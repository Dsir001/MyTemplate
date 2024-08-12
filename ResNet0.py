# -*- coding: utf-8 -*-
# @Time     :   2024/07/21 21:53:17
# @Author   :   
# @Location :   
# @File     :   ResNet.py

# Here is the code ：

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from src.efficient_kan.kan import KAN
from Transformer import *


class BasicBlock(nn.Module):      # 左侧的 residual block 结构（18-layer、34-layer）
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):      # 两层卷积 Conv2d + Shutcuts
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):      # 右侧的 residual block 结构（50-layer、101-layer、152-layer）
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):      # 三层卷积 Conv2d + Shutcuts
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:      # Shutcuts用于构建 Conv Block 和 Identity Block
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, include_to_kan:bool = False,num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7,stride=2, padding=3, bias=False)# conv1
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.bn2  = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)       # conv2_x
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)      # conv3_x
        self.transformer = VisionTransformer(
            in_shape=(block.expansion*128,28,28),
            patch_size=(4,4),
            num_heads=8,
            d_MLP=512,
            dropout=0.2,
            num_layers=8,
            classification=False
        )
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)      # conv4_x
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)      # conv5_x
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = KAN([512*block.expansion,64,num_classes]) if include_to_kan else nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(0.3)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode = 'fan_out',nonlinearity='relu')
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.maxpool(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.transformer(x)
        x = self.layer3(x)
        x = self.dropout(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        out = F.softmax(x,dim=1)
        return out


def ResNet18(num_classes,include_to_kan):
    return ResNet(BasicBlock, [2, 2, 2, 2],include_to_kan,num_classes)


def ResNet34(num_classes,include_to_kan):
    return ResNet(BasicBlock, [3, 4, 6, 3],include_to_kan,num_classes)


def ResNet50(num_classes,include_to_kan):
    return ResNet(Bottleneck, [3, 4, 6, 3],include_to_kan,num_classes)


def ResNet101(num_classes,include_to_kan):
    return ResNet(Bottleneck, [3, 4, 23, 3],include_to_kan,num_classes)


def ResNet152(num_classes,include_to_kan):
    return ResNet(Bottleneck, [3, 8, 36, 3],include_to_kan,num_classes)


def test():
    net = ResNet34(num_classes=10,include_to_kan=True).cuda()
    x  = torch.randn(64, 1, 224, 224).cuda()
    y = net(x)
    print(y.size())
    summary(net, input_size=(64, 1, 224, 224))


if __name__ == '__main__':
    net = ResNet34(num_classes=10,include_to_kan=True).cuda()
    x = torch.randn(64, 1, 224, 224).cuda()
    y = net(x)
    summary(net,input_size=(1,224,224))