import torch
import torch.nn as nn

class GroupedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.grouped_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias
        )
    
    def forward(self, x):
        return self.grouped_conv(x)
