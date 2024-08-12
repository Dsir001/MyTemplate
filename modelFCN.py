# -*- coding: utf-8 -*-
# @Time : 2024/06/29/0029 10:36
# @Author : rainbow
# @Location: hennanpuyang
# @File : model1.py


import torch
import torch.nn as nn

from torchsummary import summary
class conv(nn.Module):
    def __init__(self,in_channels,out_channels,k_s = 3):
        super().__init__()
        self.conv1  =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_s, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv1(x)
class FCN(nn.Module):
    def __init__(self):
        super(FCN,self).__init__()
        self.conv1 = conv(3,64)
        self.conv2 = conv(64,64)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,1))
        self.norm1 =nn.BatchNorm2d(64)
        self.conv3 = conv(64, 128)
        self.conv4 = conv(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,1))
        self.norm2 = nn.BatchNorm2d(128)
        self.conv5 = conv(128, 256)
        self.conv6 = conv(256, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=(4,1))
        self.norm3 = nn.BatchNorm2d(256)
        self.conv7 = conv(256, 512)
        self.conv8 = conv(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 3))
        self.norm4 = nn.BatchNorm2d(512)
        self.conv9  = conv(512,1024)
        self.conv10  =conv(1024,1024)##  *8

        self.up1 = nn.UpsamplingNearest2d(scale_factor=(2, 4))
        self.norm5 = nn.BatchNorm2d(1024)
        self.conv11= conv(1024,512)
        self.conv12 = conv(512,512)##   *2
        self.up2 = nn.UpsamplingNearest2d(scale_factor=(2, 2))
        self.norm6 = nn.BatchNorm2d(512)
        self.conv13 = conv(512,256)
        self.conv14 = conv(256,256)##   *2
        self.up3 = nn.UpsamplingNearest2d(scale_factor=(2, 2))
        self.norm7 = nn.BatchNorm2d(256)
        self.conv15 = conv(256, 128)
        self.conv16 = conv(128, 128)##   *2
        self.up4 = nn.UpsamplingNearest2d(scale_factor=(2, 2))
        self.norm8 = nn.BatchNorm2d(128)
        self.conv17 = conv(128, 64)
        self.conv18 = conv(64, 64)  ##   *2
        self.conv19 = conv(64,1)
        self.conv20  = nn.Conv2d(1,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.norm9 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.norm2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.norm3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)
        x = self.norm4(x)
        x = self.conv9(x)
        for i in range(8):
            x = self.conv10(x)
        x = self.up1(x)
        x = self.norm5(x)
        x = self.conv11(x)
        x = self.conv12(self.conv12(x))
        x = self.up2(x)
        x = self.norm6(x)
        x = self.conv13(x)
        x = self.conv14(self.conv14(x))
        x = self.up3(x)
        x = self.norm7(x)
        x = self.conv15(x)
        x = self.conv16(self.conv16(x))
        x = self.up4(x)
        x = self.norm8(x)
        x = self.conv17(x)
        x = self.conv18(self.conv18(x))
        x = self.conv20(self.conv19(x))
        x = self.norm9(x)
        x = self.sigmoid(x)
        return x
if __name__ == '__main__':
    input =  torch.randn(5,3,2048,12).cuda()
    model = FCN().cuda()
    summary(model,(3,2048,12))
    output  = model(input)
    print(output.shape)
