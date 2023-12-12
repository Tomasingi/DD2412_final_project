import torch.nn.functional as F
from torch import nn


class PackedLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha, gamma, n_estimators,
                 device=None, dtype=None):
        super().__init__()

        extended_in_features = in_features * alpha
        extended_out_features = out_features * n_estimators

        groups = gamma * n_estimators

        if extended_in_features % groups or extended_out_features % groups:
            raise ValueError("Input / output features must be divisible by groups")

        self.conv1x1 = nn.Conv1d(
            in_channels=extended_in_features,
            out_channels=extended_out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=True,
            device=device,
            dtype=dtype
        )

    def forward(self, input):
        return self.conv1x1(input)


class PackedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha, gamma,
        n_estimators, stride=1, padding=1, first=False, last=False, bias=True):

        super().__init__()

        self.n_estimators = n_estimators

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
            bias=bias,
        )

    def forward(self, input):
        return self.conv(input)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, alpha, gamma, n_estimators, downsample):
        super(BasicBlock, self).__init__()
        self.stride = stride

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(in_channels, out_channels, kernel_size, alpha, 1, n_estimators, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * alpha)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = PackedConv2d(out_channels, out_channels, kernel_size, alpha, gamma, n_estimators, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * alpha)

        self.downsample = nn.Identity()
        # if stride != 1 or in_channels != out_channels:
        if downsample:
            self.downsample = nn.Sequential(
                PackedConv2d(in_channels, out_channels, 1, alpha, gamma, n_estimators, stride=stride, padding=0, bias=False),
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
    def __init__(self, n_estimators, alpha=2, gamma=1):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.n_estimators = n_estimators

        self.in_channels = 64

        self.conv1 = PackedConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            alpha=alpha,
            gamma=1,
            n_estimators=n_estimators,
            first=True,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64 * alpha)

        self.relu = nn.ReLU(inplace=True)

        # self.layer1 = self.make_layer(64, 1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, 3, 1, alpha, gamma, n_estimators, downsample=False),
            BasicBlock(64, 64, 3, 1, alpha, gamma, n_estimators, downsample=False)
        )
        self.layer2 = self.make_layer(128, 2, downsample=True)
        self.layer3 = self.make_layer(256, 2, downsample=True)
        self.layer4 = self.make_layer(512, 2, downsample=True)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten1 = nn.Flatten(2)

        self.linear = PackedLinear(512, 10, alpha, 1, n_estimators)
        self.flatten2 = nn.Flatten(1)

    def make_layer(self, out_channels, stride, downsample=False):
        l1 = BasicBlock(self.in_channels, out_channels, 3, stride, self.alpha, self.gamma, self.n_estimators, downsample=downsample)
        self.in_channels = out_channels

        l2 = BasicBlock(self.in_channels, out_channels, 3, 1, self.alpha, self.gamma, self.n_estimators, downsample=False)
        self.in_channels = out_channels
        return nn.Sequential(l1, l2)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.flatten1(out)
        out = self.linear(out)
        out = self.flatten2(out)
        return out