# -*- coding: utf-8 -*-
# @Time     :   2024/07/28 15:46:49
# @Author   :   dcj
# @Location :   hennanpuyang
# @File     :   muResNet.py


import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from torchinfo import summary
from Transformer import VisionTransformer


class BasicBlock_(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=(1,1),Upsampling_size = None):    
        super(BasicBlock_, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.up  = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=Upsampling_size),
            nn.BatchNorm2d(in_planes)
        ) if Upsampling_size is not None else nn.Sequential() 
        self.shortcut = nn.Sequential()
        if (stride != 1 and stride!=(1,1)) or in_planes != self.expansion*planes:      
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x =self.up(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class Bottleneck_(nn.Module):     
    expansion = 4

    def __init__(self, in_planes, planes, stride=1,Upsampling_size = None):      
        super(Bottleneck_, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=Upsampling_size),
            nn.BatchNorm2d(in_planes)
        ) if Upsampling_size is not None else nn.Sequential() 
        self.shortcut = nn.Sequential()
        if (stride != 1 and stride!=(1,1)) or in_planes != self.expansion*planes:      
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        x = self.up(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResFCNNet(nn.Module):
    def __init__(self,block,num_blocks,toViT=False):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.toVit = toViT
        self.channels = 64
        self.encoder1 = self._make_layer(block,64,num_blocks[0],2)
        self.encoder2 = self._make_layer(block,128,num_blocks[1],2)
        self.encoder3 = self._make_layer(block,256,num_blocks[2],2)
        self.transfomer = VisionTransformer(
            in_shape=(self.channels,14,14),
            patch_size=(1,1),
            num_heads=8,
            d_MLP=1024,
            dropout=0.1,
            num_layers=12,
            classification=False
        ) if toViT else nn.Sequential()
        self.decoder1 = self._make_layer(block,256,num_blocks[3],1,2)
        self.decoder2 = self._make_layer(block,128,num_blocks[4],1,2)
        self.decoder3 = self._make_layer(block,64,num_blocks[5],1,2)
        self.finalconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(self.channels),
            nn.Conv2d(self.channels,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def _make_layer(self,block,in_channels,num_blocks,stride=1,Upsampling_size = None):
        stride = [stride]+[1]*(num_blocks-1)
        Upsampling_size = [Upsampling_size]+[None]*(num_blocks-1)
        layer = []
        for stride,up_size in zip(stride,Upsampling_size):
            layer.append(block(self.channels,in_channels,stride,up_size))
            self.channels = in_channels*block.expansion

        return nn.Sequential(*layer)
    def forward(self,x):
        x = self.firstconv(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        print(x.shape)
        b,c,h,w = x.size()
        x = self.transfomer(x).transpose(1,2).contiguous().view(b,c,h,w)if self.toVit else self.transfomer(x)
        print(x.shape)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.finalconv(x)

        return x
if __name__=='__main__':
    # a = torch.randn(6,3,224,224).cuda()
    # model = ResFCNNet(BasicBlock_,num_blocks=[2,2,12,12,2,2],toViT=True).cuda()
    # summary(model,(1,3,224,224))
    # print(model(a).shape)
    lastmodel_path ='1' 
    print(f'The last Model has been saved \'{lastmodel_path}\'!!')