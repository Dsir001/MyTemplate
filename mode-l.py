# -*- coding: utf-8 -*-
# @Time : 2024/07/14/0014 15:08
# @Author : rainbow
# @Location: henan
# @File : model.py


import torch
import torch.nn as nn


class resnet_block(nn.Module):
    def __init__(self,in_channels):
        super(resnet_block, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU()
    def forward(self,x):
        x_ = self.doubleconv(x)
        x=x+x_
        x = self.relu(x)
        return x
class conv(nn.Module):
    def __init__(self,in_channels,out_channels,k_s = 3,padding=1, stride=1):
        super().__init__()
        self.conv1  =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_s, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv1(x)
class ResNet(nn.Module):
    def __init__(self,in_channel,num_resnet_block : int= 6):
        super(ResNet,self).__init__()
        self.first_conv = conv(in_channels=in_channel,out_channels=64,k_s=3,padding=1,stride=1)
        self.resnet  = resnet_block(in_channels=64)
        self.num = num_resnet_block
    def forward(self,x):
        x = self.first_conv(x)
        for i in range(self.num):
            x  = self.resnet(x)
        return x
class Maxpool(nn.Module):
    def __init__(self,in_channels,out_channels,maxpool_size):
        super(Maxpool,self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_size),
            nn.BatchNorm2d(in_channels)
        )
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        x = self.maxpool(x)
        x = self.doubleconv(x)
        return x
class Upsampling(nn.Module):
    def __init__(self,in_channels,out_channels,upsampling_size):
        super(Upsampling,self).__init__()
        self.upsampling = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=upsampling_size),
            nn.BatchNorm2d(in_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x,x_cat):
        x = self.upsampling(x)
        x  =self.conv(x)
        if x_cat is not None:
            x = torch.cat([x,x_cat],dim=1)
        x = self.doubleconv(x)
        return x
class Unet(nn.Module):
    def __init__(self,in_channel):
        super(Unet,self).__init__()
        self.first_conv = conv(in_channels=in_channel, out_channels=64, k_s=3, padding=1, stride=1)
        self.second_conv = conv(in_channels=64, out_channels=64, k_s=3, padding=1, stride=1)
        self.maxpool1 = Maxpool(in_channels=64, out_channels=128, maxpool_size=(2, 2))
        self.maxpool2 = Maxpool(in_channels=128, out_channels=256, maxpool_size=(2, 2))
        self.upsampling2 = Upsampling(in_channels=256, out_channels=128, upsampling_size=(2, 2))
        self.upsampling3 = Upsampling(in_channels=128, out_channels=64, upsampling_size=(2, 2))
    def forward(self,x):
        x = self.first_conv(x)
        x1 = self.second_conv(x)
        x2 = self.maxpool1(x1)
        x = self.maxpool2(x2)
        x = self.upsampling2(x,x2)
        x = self.upsampling3(x,x1)
        return x
class CNN(nn.Module):
    def __init__(self,in_channel,num_class:int=10):
        super(CNN,self).__init__()
        self.cnn = nn.Sequential(
            conv(in_channels=in_channel, out_channels=128, k_s=3, stride=1, padding=1),
            conv(in_channels=128, out_channels=128, k_s=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            conv(in_channels=128, out_channels=64, k_s=3, stride=1, padding=1),
            conv(in_channels=64, out_channels=64, k_s=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            conv(in_channels=64, out_channels=16, k_s=3, stride=1, padding=1),
            conv(in_channels=16, out_channels=16, k_s=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            conv(in_channels=16, out_channels=1, k_s=3, stride=1, padding=1),
            conv(in_channels=1, out_channels=1, k_s=3, stride=1, padding=1),
        )
        self.linear  = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_class),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = self.cnn(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.resnet = ResNet(in_channel=1,num_resnet_block= 3)
        self.unet  = Unet(in_channel=1)
        self.cnn = CNN(in_channel=128,num_class=10)
    def forward(self,x):
        x1 = self.resnet(x)
        x2 = self.unet(x)
        x = self.cnn(torch.cat([x1,x2],dim=1))
        return x
if __name__ == "__main__":
    x = torch.randn(8,1,224,224)
    model =Net()
    pred = model(x)
    
    print(pred.shape) 