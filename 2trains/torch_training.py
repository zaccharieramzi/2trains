import time

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
import torch
from torch.optim import SGD
from torchvision import datasets, models, transforms
from tqdm import tqdm


seed_everything(0, workers=True)
# unused trainer to replicate the benchopt setting
trainer = Trainer(accelerator=None)
## Data ##
# Get the cifar dataset, normalize it and augment it
normalization_mean = (0.4914, 0.4822, 0.4465)
normalization_std = (0.2023, 0.1994, 0.2010)
transform_list = [
    transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(
        normalization_mean,
        normalization_std,
    )
]
transform = transforms.Compose(transform_list)
ds = datasets.CIFAR10(
    root='./data',
    download=True,
    transform=transform,
    train=True,
)
batch_size = 128
dataloader = torch.utils.data.DataLoader(
    ds, batch_size=batch_size,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True, shuffle=True
)

## Model ##
def remove_initial_downsample(large_model):
    large_model.conv1 = torch.nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    large_model.maxpool = torch.nn.Identity()
    return large_model

model = models.resnet18(num_classes=10)
# we fit the resnet18 model to the cifar dataset
model = remove_initial_downsample(model)

## Training preparation ##
n_epochs = 200
optimizer = SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    nesterov=False,
    weight_decay=0.0005,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=n_epochs,
)
criterion = torch.nn.CrossEntropyLoss()
if torch.cuda.is_available():
    model.cuda()
    criterion

## Training loop ##

# warm-up
for _ in range(5):
    X = next(iter(dataloader))[0]
    if torch.cuda.is_available():
        X = X.cuda()
    model(X)
# actual timing
timing_epochs = 1
start = time.time()
for _ in range(min(timing_epochs, n_epochs)):
    for X, y in tqdm(dataloader):
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()

        optimizer.step()
    scheduler.step()
end = time.time()
print('Training took {:.2f} seconds'.format(end - start))
print('This gives a per epoch cost of {:.2f} seconds'.format((end - start) / timing_epochs))
