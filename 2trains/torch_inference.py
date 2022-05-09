import time

from pytorch_lightning.utilities.seed import seed_everything
import torch
from torch.optim import SGD
from torchvision import datasets, models, transforms
from tqdm import tqdm


seed_everything(0, workers=True)
torch.backends.cudnn.benchmark = True
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

## Training loop ##
if torch.cuda.is_available():
    model.cuda()
model.eval()
# warm-up
with torch.no_grad():
    for _ in range(5):
        X = next(iter(dataloader))[0]
        if torch.cuda.is_available():
            X = X.cuda()
        model(X)
# actual timing
timing_epochs = 1
start = time.time()
with torch.no_grad():
    for X, y in tqdm(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
        output = model(X)
        if torch.cuda.is_available():
            output = output.cpu()
        _ = output.detach().numpy().item()
end = time.time()
print('Training took {:.2f} seconds'.format(end - start))
print('This gives a per epoch cost of {:.2f} seconds'.format((end - start) / timing_epochs))
