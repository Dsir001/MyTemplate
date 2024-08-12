# -*- coding: utf-8 -*-
# @Time : 2024/06/16/0016 16:07
# @Author : rainbow
# @Location: 江西
# @File : trainmodel.py


import torch
import torch.nn as nn
from torchsummary import summary


class downsample(nn.Module):
    def __init__(self,in_channels,out_channels,maxpool_size = None):
        super(downsample, self).__init__()
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3,stride=1,padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_size,stride=maxpool_size),
            nn.BatchNorm2d(in_channels)
        )
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1,stride=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,input_x):
        input_x  = self.conv_pool(input_x)
        input_x = self.conv1(input_x)
        input_x = self.norm(input_x)
        input_x = self.relu(input_x)
        input_x  =self.conv2(input_x)
        input_x = self.norm(input_x)
        input_x = self.relu(input_x)
        return input_x
class upsample(nn.Module):
    def __init__(self,in_channels,out_channels ,upsampling_size = (2,2)):
        super(upsample,self).__init__()
        self.conv_up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=upsampling_size),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1,stride=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self,input_x,x_cat = None):
        input_x = self.conv_up(input_x)
        if input_x is not None:
            input_x = torch.cat([x_cat,input_x],dim = 1)
        input_x = self.conv1(input_x)
        input_x = self.norm(input_x)
        input_x = self.relu(input_x)
        input_x = self.conv2(input_x)
        input_x = self.norm(input_x)
        input_x = self.relu(input_x)
        return input_x

class resconcate(nn.Module):
    def __init__(self,in_channels):
        super(resconcate,self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    def forward(self,input_x):
        x = input_x
        input_x = self.doubleconv(input_x)
        input_x = input_x+x
        input_x = self.norm(input_x)
        input_x = self.relu(input_x)
        return input_x
class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet,self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.downsample1 = downsample(in_channels=64,  out_channels=128, maxpool_size=(2, 1))
        self.downsample2 = downsample(in_channels=128, out_channels=256, maxpool_size=(2, 1))
        self.downsample3 = downsample(in_channels=256, out_channels=512, maxpool_size=(2, 1))
        self.downsample4 = downsample(in_channels=512, out_channels=1024, maxpool_size=(2, 3))
        self.conv1 = nn.Conv2d(1024,1024,kernel_size=3,padding=1,stride=1,bias=False)##*6
        self.norm1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()
        self.upsample1 = upsample(in_channels=1024, out_channels=512, upsampling_size=(1, 2))
        self.upsample2 = upsample(in_channels=512, out_channels=256, upsampling_size=(1, 4))
        self.upsample3 = upsample(in_channels=256, out_channels=128, upsampling_size=(1, 2))
        self.upsample4 = upsample(in_channels=128, out_channels=64, upsampling_size=(1, 2))
        self.up = nn.UpsamplingNearest2d(scale_factor=(2,1))
        self.norm2  = nn.BatchNorm2d(64)
        self.conv2  = nn.Conv2d(64,64,kernel_size=3,padding=1,stride=1,bias=False)#*2
        self.norm3  = nn.BatchNorm2d(64)
        self.relu2  = nn.ReLU()
        self.conv3  = nn.Conv2d(64,1,kernel_size=3,padding=1,stride=1,bias=False)
        self.norm4 = nn.BatchNorm2d(1)
        self.relu3 = nn.ReLU()
        self.conv4  = nn.Conv2d(1,1,kernel_size=3,padding=1,stride=1,bias=False)
        self.norm5 = nn.BatchNorm2d(1)
        self.relu4 = nn.Sigmoid()
        self.resconv1 = nn.Conv2d(96,64,kernel_size=3,padding=1,stride=1,bias=False)
        self.resnorm1 = nn.BatchNorm2d(64)
        self.resrelu1 = nn.ReLU()
        self.resconcate1 = resconcate(in_channels=64)
        self.resconv2 = nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=1, bias=False)
        self.resnorm2 = nn.BatchNorm2d(128)
        self.resrelu2 = nn.ReLU()
        self.resconcate2 = resconcate(in_channels=128)
        self.resconv3 = nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1, bias=False)
        self.resnorm3 = nn.BatchNorm2d(256)
        self.resrelu3 = nn.ReLU()
        self.resconcate3 = resconcate(in_channels=256)
        self.resconv4 = nn.Conv2d(1536, 512, kernel_size=3, padding=1, stride=1, bias=False)
        self.resnorm4 = nn.BatchNorm2d(512)
        self.resrelu4 = nn.ReLU()
        self.resconcate4 = resconcate(in_channels=512)
        # self.dropout = nn.Dropout(0.6)
        # self.crop1 = nn.
    def forward(self,x):
        batch_size = x.shape[0]
        x1 = self.doubleconv(x)
        x2  = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x = self.downsample4(x4)
        for i in range(6):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu1(x)
        # x  = self.dropout(x)

        x4 = self.resconv4(x4.view(batch_size,1536,128,8))
        x4 = self.resnorm4(x4)
        x4 = self.resrelu4(x4)
        for i in range(3):
            x4 = self.resconcate4(x4)
        # x4  = self.dropout(x4)
        x = self.upsample1(x,x4)

        x3 = self.resconv3(x3.view(batch_size, 384, 128, 32))
        x3 = self.resnorm3(x3)
        x3 = self.resrelu3(x3)
        for i in range(3):
            x3 = self.resconcate3(x3)
        # x3  = self.dropout(x3)
        x = self.upsample2(x, x3)

        x2 = self.resconv2(x2.view(batch_size,192,128,64))
        x2 = self.resnorm2(x2)
        x2 = self.resrelu2(x2)
        for i in range(3):
            x2 = self.resconcate2(x2)
        # x2  = self.dropout(x2)
        x = self.upsample3(x,x2)

        x1 = self.resconv1(x1.view(batch_size,96,128,128))
        x1 = self.resnorm1(x1)
        x1 = self.resrelu1(x1)
        for i in range(3):
            x1 = self.resconcate1(x1)
        # x1  = self.dropout(x1)
        x = self.upsample4(x,x1)

        x = self.up(x)
        x = self.norm2(x)
        # x = self.dropout(x)
        for i in range(2):
            x = self.conv2(x)
            x = self.norm3(x)
            x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm4(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm5(x)
        x = self.relu4(x)

        return x





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(5, 3, 2048, 12).to(device)
    x_cat = torch.randn(5, 512, 128, 8).to(device)
    # model  = downsample(in_channels=64,out_channels=128,maxpool_size=(2,1)).to(device)
    # model  = upsample(in_channels=1024,out_channels=512,upsampling_size=(1,2)).to(device)
    # model = resconcate(in_channels=64).to(device)
    model = ResUNet().to(device)

    pred = model(x)
    # pred = x.view(5,-1,128,8)
    print(pred.shape)
    print(torch.max(pred))
    summary(model, input_size=(3,2048, 12))