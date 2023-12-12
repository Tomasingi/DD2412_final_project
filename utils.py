import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

import baseline_models
import packed_models

class HParams:
    def __init__(self):
        self.epochs = 75
        self.batch_size = 128
        self.lr = 0.05
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.gamma_lr = 0.1
        self.milestones = [25, 50]
        self.data_transforms_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.data_transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data(hparams, training=True):
    """
    Copied from:
    https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/
    """
    if training:
        # CIFAR10 training dataset.
        dataset_train = datasets.CIFAR10(
            root='data',
            train=True,
            download=True,
            transform=hparams.data_transforms_train,
        )
    # CIFAR10 validation dataset.
    dataset_valid = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=hparams.data_transforms_test
    )
    train_loader = None
    if training:
        # Create data loaders.
        train_loader = DataLoader(
            dataset_train,
            batch_size=hparams.batch_size,
            shuffle=True
        )
    valid_loader = DataLoader(
        dataset_valid,
        batch_size=hparams.batch_size,
        shuffle=False
    )
    return train_loader, valid_loader

def show_example(data_loader, idx=None):
    # Show one example image
    if idx is None:
        idx = np.random.randint(0, len(data_loader.dataset))
    img, label = data_loader.dataset[idx]
    temp_img = np.transpose(img, (1, 2, 0))
    plt.imshow(temp_img, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()

def get_model(name):
    if name == 'baseline':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        model.fc = nn.Linear(512, 10)
        return model
    if name == 'baseline_scratch':
        return baseline_models.ResNet18()
    if name == 'packed':
        return packed_models.PackedResNet18(alpha=2, gamma=2, n_estimators=4)
    raise ValueError(f'Invalid model name: {name}')