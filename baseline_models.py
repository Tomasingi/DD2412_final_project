import torch.nn.functional as F
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1)
        )

        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, 2,
                       downsample=nn.Sequential(
                           nn.Conv2d(
                               in_channels=64,
                               out_channels=128,
                               kernel_size=1,
                               stride=2,
                               bias=False
                           ),
                           nn.BatchNorm2d(128)
                       )
            ),
            BasicBlock(128, 128, 1)
        )

        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, 2,
                       downsample=nn.Sequential(
                         nn.Conv2d(
                              in_channels=128,
                              out_channels=256,
                              kernel_size=1,
                              stride=2,
                              bias=False
                         ),
                         nn.BatchNorm2d(256)
                       )
            ),
            BasicBlock(256, 256, 1)
        )

        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, 2,
                       downsample=nn.Sequential(
                         nn.Conv2d(
                              in_channels=256,
                              out_channels=512,
                              kernel_size=1,
                              stride=2,
                              bias=False
                         ),
                         nn.BatchNorm2d(512)
                       )
            ),
            BasicBlock(512, 512, 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = nn.Flatten(1)(out)
        out = self.fc(out)

        return out