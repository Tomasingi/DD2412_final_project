from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_data(batch_size=64):
    """
    Copied from:
    https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/
    """

    # CIFAR10 training dataset.
    dataset_train = datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )
    # CIFAR10 validation dataset.
    dataset_valid = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )
    # Create data loaders.
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader