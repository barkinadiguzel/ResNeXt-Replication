import torch
import torch.nn as nn
from layers.conv_layer import ConvLayer
from layers.grouped_conv import GroupedConv

class ResNeXtBottleneck(nn.Module):
    expansion = 4  # same as ResNet bottleneck (out_channels = mid_channels * expansion)

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4):
        super().__init__()
        D = int(cardinality * base_width)
        mid_channels = D  

        # 1x1 conv: reduce
        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid_channels)

        # 3x3 grouped conv: transform (groups = cardinality)
        self.conv_group = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                                    stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn_group = nn.BatchNorm2d(mid_channels)

        # 1x1 conv: expand
        self.conv_expand = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut: identity or projection when shape changes
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv_reduce(x)
        out = self.bn_reduce(out)
        out = self.relu(out)

        out = self.conv_group(out)
        out = self.bn_group(out)
        out = self.relu(out)

        out = self.conv_expand(out)
        out = self.bn_expand(out)

        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
