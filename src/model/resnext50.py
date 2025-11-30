import torch
import torch.nn as nn
from blocks.resnext_block import ResNeXtBottleneck

class ResNeXt50(nn.Module):
    def __init__(self, num_classes=1000, cardinality=32, bottleneck_width=4):
        super(ResNeXt50, self).__init__()
        
        # Stem: initial conv + maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Conv2_x stage
        self.stage2 = nn.Sequential(
            ResNeXtBottleneck(64, 256, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(256, 256, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(256, 256, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
        )
        
        # Conv3_x stage
        self.stage3 = nn.Sequential(
            ResNeXtBottleneck(256, 512, stride=2, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(512, 512, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(512, 512, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(512, 512, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
        )
        
        # Conv4_x stage
        self.stage4 = nn.Sequential(
            ResNeXtBottleneck(512, 1024, stride=2, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(1024, 1024, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(1024, 1024, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(1024, 1024, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(1024, 1024, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(1024, 1024, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
        )
        
        # Conv5_x stage
        self.stage5 = nn.Sequential(
            ResNeXtBottleneck(1024, 2048, stride=2, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(2048, 2048, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
            ResNeXtBottleneck(2048, 2048, stride=1, cardinality=cardinality, bottleneck_width=bottleneck_width),
        )
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
