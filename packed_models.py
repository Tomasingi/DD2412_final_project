import torch.nn.functional as F
from torch import nn

class PackedFinalLinear(nn.Module):
    def __init__(self, in_channels, out_channels, alpha, gamma, n_estimators):
        super().__init__()

        real_in = in_channels * alpha
        real_out = out_channels * n_estimators

        groups = gamma * n_estimators

        if real_in % groups or real_out % groups:
            raise ValueError('Input / output features must be divisible by groups')

        self.conv = nn.Conv1d(
            in_channels=real_in,
            out_channels=real_out,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=True
        )

    def forward(self, x):
        return self.conv(x)


class PackedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, alpha, gamma, n_estimators, first=False, last=False, bias=True):
        super().__init__()

        real_in = in_channels * alpha
        real_out = out_channels * alpha
        if first:
            real_in = in_channels
        if last:
            real_out = out_channels * n_estimators

        groups = gamma * n_estimators
        if first:
            groups = 1

        if real_in % groups or real_out % groups:
            raise ValueError('Input / output features must be divisible by groups')

        self.conv = nn.Conv2d(
            in_channels=real_in,
            out_channels=real_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class PackedBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, alpha, gamma, n_estimators, downsample=False):
        super().__init__()

        self.conv1 = PackedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            alpha=alpha,
            gamma=1,
            n_estimators=n_estimators,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels * alpha)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = PackedConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            alpha=alpha,
            gamma=gamma,
            n_estimators=n_estimators,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels * alpha)

        self.downsample = nn.Identity()
        if downsample:
            self.downsample = nn.Sequential(
                PackedConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    alpha=alpha,
                    gamma=gamma,
                    n_estimators=n_estimators,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * alpha)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        x = self.downsample(x)

        out += x
        out = self.relu(out)

        return out

class PackedResNet18(nn.Module):
    def __init__(self, alpha, gamma, n_estimators):
        super().__init__()

        self.conv1 = PackedConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            alpha=alpha,
            gamma=1,
            n_estimators=n_estimators,
            first=True,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64 * alpha)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = nn.Sequential(
            PackedBasicBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators
            ),
            PackedBasicBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators
            )
        )

        self.layer2 = nn.Sequential(
            PackedBasicBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators,
                downsample=True
            ),
            PackedBasicBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators
            )
        )

        self.layer3 = nn.Sequential(
            PackedBasicBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators,
                downsample=True
            ),
            PackedBasicBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators
            )
        )

        self.layer4 = nn.Sequential(
            PackedBasicBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators,
                downsample=True
            ),
            PackedBasicBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1,
                alpha=alpha,
                gamma=gamma,
                n_estimators=n_estimators
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = PackedFinalLinear(512, 10, alpha, 1, n_estimators)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)

        out = nn.Flatten(2)(out)
        out = self.fc(out)
        out = nn.Flatten(1)(out)

        return out