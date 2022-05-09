import os
import time

os.environ['TF_CUDNN_USE_AUTOTUNE'] = "1"
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

from keras.applications.resnet import ResNet
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow_addons.optimizers import extend_with_decoupled_weight_decay
import tensorflow_datasets as tfds
from tqdm import tqdm


tf.random.set_seed(0)
## Data ##
# Get the cifar dataset, normalize it and augment it
normalization_mean = (0.4914, 0.4822, 0.4465)
normalization_std = (0.2023, 0.1994, 0.2010)
data_aug_layer = tf.keras.models.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=4),
    tf.keras.layers.RandomCrop(height=32, width=32),
    tf.keras.layers.RandomFlip('horizontal'),
])
normalization = tf.keras.layers.Normalization(
    mean=normalization_mean,
    variance=np.square(normalization_std),
)
batch_size = 128
ds = tfds.load(
    'cifar10',
    split='train',
    as_supervised=True,
).map(
    lambda x, y: (data_aug_layer(x[None], training=True)[0], y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).shuffle(
    buffer_size=1000,  # For now a hardcoded value
    reshuffle_each_iteration=True,
).batch(
    batch_size,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).map(
    lambda x, y: (normalization(x/255), y),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE,
)

## Model ##
def basic_block(x, filters, stride=1, use_bias=True, conv_shortcut=True,
                name=None):
    """A basic residual block for ResNet18 and 34.

    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.

    Returns:
    Output tensor for the basic residual block.
    """
    bn_axis = 3
    kernel_size = 3

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=use_bias,
            name=name + '_0_conv',
            kernel_initializer='he_normal',
        )(x)
        shortcut = layers.BatchNormalization(
            momentum=0.9,
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    if stride > 1:
        x = layers.ZeroPadding2D(
            padding=((1, 0), (1, 0)),
            name=name + '_1_pad',
        )(x)
        padding_mode = 'valid'
    else:
        padding_mode = 'same'
    x = layers.Conv2D(
        filters, kernel_size, padding=padding_mode, strides=stride,
        kernel_initializer='he_normal',
        use_bias=use_bias,
        name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding='SAME',
        use_bias=use_bias,
        kernel_initializer='he_normal',
        name=name + '_2_conv',
    )(x)
    x = layers.BatchNormalization(
        momentum=0.9,
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x

def stack_block(
    x,
    filters,
    n_blocks,
    block_fn,
    stride1=2,
    first_shortcut=True,
    name=None,
    use_bias=False,
):
    """A set of stacked residual blocks.

    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    n_blocks: integer, blocks in the stacked blocks.
    block_fn: callable, function defining one block.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.

    Returns:
    Output tensor for the stacked basic blocks.
    """
    x = block_fn(
        x,
        filters,
        stride=stride1,
        conv_shortcut=first_shortcut,
        use_bias=use_bias,
        name=name + '_block1',
    )
    for i in range(2, n_blocks + 1):
        x = block_fn(
            x,
            filters,
            conv_shortcut=False,
            use_bias=use_bias,
            name=name + '_block' + str(i),
        )
    return x


def remove_initial_downsample(large_model, use_bias=False):
    trimmed_model = models.Model(
        inputs=large_model.get_layer('conv2_block1_1_conv').input,
        outputs=large_model.outputs,
    )
    first_conv = layers.Conv2D(
        64,
        3,
        activation='linear',
        padding='same',
        use_bias=use_bias,
        kernel_initializer='he_normal',
        name='conv1_conv',
    )
    input_shape = list(large_model.input_shape[1:])
    input_shape[0] = input_shape[0] // 4
    input_shape[1] = input_shape[1] // 4
    small_model = models.Sequential([
        layers.Input(input_shape),
        first_conv,
        layers.BatchNormalization(
            momentum=0.9,
            axis=-1, epsilon=1.001e-5, name='conv1_bn'),
        layers.Activation('relu', name='conv1_relu'),
        trimmed_model,
    ])
    return small_model


def ResNet18(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             use_bias=True,
             no_initial_downsample=False,
             **kwargs):
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = stack_block(
            x,
            64,
            2,
            basic_block,
            use_bias=use_bias,
            first_shortcut=False,
            stride1=1,
            name='conv2',
        )
        x = stack_block(
            x,
            128,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv3',
        )
        x = stack_block(
            x,
            256,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv4',
        )
        return stack_block(
            x,
            512,
            2,
            basic_block,
            use_bias=use_bias,
            name='conv5',
        )

    model = ResNet(
        stack_fn,
        False,
        use_bias,
        'resnet18',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs,
    )
    if no_initial_downsample:
        model = remove_initial_downsample(model, use_bias=use_bias)
    return model

model = ResNet18(
    weights=None,
    classes=10,
    classifier_activation='softmax',
    input_shape=(4*32, 4*32, 3),
    no_initial_downsample=True,  # to fit the resnet to the cifar setting
    use_bias=False,
)

## Training preparation ##
n_epochs = 200
decoupled_wd = 0.0005
lr = 0.1

optimizer = extend_with_decoupled_weight_decay(
    tf.keras.optimizers.SGD,
)(
    weight_decay=decoupled_wd*lr,
    lr=lr,
    momentum=0.9,
    nesterov=False,
)

class LRWDSchedulerCallback(tf.keras.callbacks.LearningRateScheduler):
    """Callback that schedules jointly the learning rate and the weight decay

    This is necessary as in TensorFlow, the decoupled weight decay is not
    multiplied by the learning rate.
    This is mentionned in the docs:
    https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/extend_with_decoupled_weight_decay
    In this case, they mention a per-step schedule, but in order to stick
    to the PyTorch way of updating LR and WD, we stick to a
    per-epoch schedule which is conveniently done with a Callback.
    """
    def __init__(self, lr_schedule, wd_schedule, verbose=0):
        super(LRWDSchedulerCallback, self).__init__(
            schedule=lr_schedule,
            verbose=verbose,
        )
        self.wd_schedule = wd_schedule

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        if not hasattr(self.model.optimizer, 'weight_decay'):
            return
        try:  # new API
            wd = float(backend.get_value(
                self.model.optimizer.weight_decay,
            ))
            wd = self.wd_schedule(epoch, wd)
        except TypeError:  # Support for old API for backward compatibility
            wd = self.wd_schedule(epoch)
        if not isinstance(wd, (tf.Tensor, float, np.float32, np.float64)):
            raise ValueError(
                'The output of the "schedule" function '
                f'should be float. Got: {wd}')
        if isinstance(wd, tf.Tensor) and not wd.dtype.is_floating:
            raise ValueError(
                'The dtype of `wd` Tensor should be float. '
                f'Got: {wd.dtype}',
            )
        backend.set_value(
            self.model.optimizer.weight_decay,
            backend.get_value(wd),
        )

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        logs['wd'] = backend.get_value(self.model.optimizer.weight_decay)

lr_scheduler, wd_scheduler = [
    tf.keras.optimizers.schedules.CosineDecay(
        value,
        n_epochs,  # the equivalent of T_max
    ) for value in [lr, decoupled_wd*lr]
]
lr_wd_cback = LRWDSchedulerCallback(
    lr_schedule=lr_scheduler,
    wd_schedule=wd_scheduler,
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics='accuracy',
    steps_per_execution=1,  # even when setting to 100, this is still slower than torch
)

cback_list = tf.keras.callbacks.CallbackList(
    [lr_wd_cback],
    model=model,
)
# It's important to create the callback list ourselves in order
# to avoid the overhead of having to store a history of the
# training and using a progressbar

## Training loop ##
# warm-up
model.fit(
    ds.take(1),
    callbacks=cback_list,
    epochs=5,
    verbose=0,
)
# actual timing
timing_epochs = 1
start = time.time()
model.fit(
    ds,
    callbacks=cback_list,
    epochs=timing_epochs,
    verbose=0,
)
end = time.time()
print('Training took {:.2f} seconds'.format(end - start))
print('This gives a per epoch cost of {:.2f} seconds'.format((end - start) / timing_epochs))
