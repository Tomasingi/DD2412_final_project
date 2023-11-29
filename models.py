import torch.nn.functional as F
from torch import nn


class PackedLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha, gamma, num_estimators,
                 bias=True, first=False, last=False, device=None, dtype=None):
        super().__init__()

        self.first = first
        self.num_estimators = num_estimators

        # Define the number of features of the underlying convolution
        extended_in_features = in_features * alpha
        extended_out_features = out_features * alpha
        if first:
            extended_in_features = in_features
        if last:
            extended_out_features = out_features * num_estimators

        # Define the number of groups of the underlying convolution
        groups = gamma * num_estimators
        if first:
            groups = 1

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
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(self, input):
        return self.conv1x1(input)


class PackedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha, gamma,
        num_estimators, stride=1, padding=1, first=False, last=False, bias=True, device=None, dtype=None):

        super().__init__()

        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = in_channels * alpha
        extended_out_channels = out_channels * alpha
        if first:
            extended_in_channels = in_channels
        if last:
            extended_out_channels = out_channels * num_estimators

        # Define the number of groups of the underlying convolution
        groups = gamma * num_estimators
        if first:
            groups = 1

        while (extended_in_channels % groups != 0
            or extended_in_channels // groups < 64) and groups >= 2 * num_estimators:
            gamma -= 1
            groups = gamma * num_estimators

        if extended_in_channels % groups or extended_out_channels % groups:
            raise ValueError("Input / output channels must be divisible by groups")

        self.conv = nn.Conv2d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=groups,
            bias=bias,
            padding_mode='zeros',
            device=device,
            dtype=dtype
        )

    def forward(self, input):
        return self.conv(input)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, alpha, gamma, num_estimators):
        super(BasicBlock, self).__init__()
        self.stride = stride

        # No subgroups for the first layer
        self.conv1 = PackedConv2d(in_planes, planes, 3, alpha, 1, num_estimators, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * alpha)
        self.conv2 = PackedConv2d(planes, planes, 3, alpha, gamma, num_estimators, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * alpha)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                PackedConv2d(in_planes, planes, 1, alpha, gamma, num_estimators, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * alpha)
            )

        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PackedResnet18(nn.Module):
    def __init__(self, num_estimators, alpha=2, gamma=1):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.num_estimators = num_estimators
        self.in_planes = 64

        self.conv1 = PackedConv2d(3, 64, 3, alpha, 1, num_estimators, stride=1, bias=False, first=True)

        self.bn1 = nn.BatchNorm2d(64 * alpha)

        self.layer1 = self.make_layer(64, 1)
        self.layer2 = self.make_layer(128, 2)
        self.layer3 = self.make_layer(256, 2)
        self.layer4 = self.make_layer(512, 2)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = PackedLinear(512, 10, alpha, 1, num_estimators, last=True)

    def make_layer(self, planes, stride):
        l1 = BasicBlock(self.in_planes, planes, stride, self.alpha, self.gamma, self.num_estimators)
        self.in_planes = planes

        l2 = BasicBlock(self.in_planes, planes, 1, self.alpha, self.gamma, self.num_estimators)
        self.in_planes = planes
        return nn.Sequential(l1, l2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = out.unsqueeze(-1)
        out = self.linear(out)
        out = out.squeeze(-1)
        return out