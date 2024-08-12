 # -*- coding: utf-8 -*-
# @Time : 2024/07/01/0001 16:51
# @Author : rainbow
# @Location: henanpuyang
# @File : RESUNET.py

import torch
import torch.nn  as nn
# from torchsummary import summary
class ResMaxpool(nn.Module):
    def __init__(self,in_channels,maxpool_size):
        super(ResMaxpool,self).__init__()
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=maxpool_size),
            nn.BatchNorm2d(in_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    def forward(self,x):
        x_pool = self.maxpool(x)
        x = self.conv(x_pool)
        x = x+x_pool
        x  = self.relu(self.norm(x))
        return x
class ResUpsampling(nn.Module):
    def __init__(self,in_channels,upsampling_size):
        super(ResUpsampling,self).__init__()
        self.upsampling = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=upsampling_size),
            nn.BatchNorm2d(in_channels)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_up = self.upsampling(x)
        x = self.conv(x_up)
        x = x + x_up
        x = self.relu(self.norm(x))
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
class res_concate(nn.Module):
    def __init__(self,in_channels,num_res):
        super(res_concate,self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.num_res = num_res
    def forward(self,x):
        for i in range(self.num_res):
            x_r = self.doubleconv(x)
            x = x+x_r
            x = self.norm(x)
            x = self.relu(x)
        return x
class Res_U_Res_Net(nn.Module):
    def __init__(self):
        super(Res_U_Res_Net,self).__init__()
        self.fisrt_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.resmaxpool = nn.Sequential(
            ResMaxpool(in_channels=64, maxpool_size=(2, 3)),
            ResMaxpool(in_channels=64, maxpool_size=(2, 1)),
            ResMaxpool(in_channels=64, maxpool_size=(2, 1))
        )
        self.resupsampling = nn.Sequential(
            ResUpsampling(in_channels=64, upsampling_size=(1, 4)),
            ResUpsampling(in_channels=64, upsampling_size=(1, 4)),
            ResUpsampling(in_channels=64, upsampling_size=(1, 2)),
        )
        self.maxpool1 = Maxpool(in_channels=64, out_channels=128, maxpool_size=(2, 2))
        self.maxpool2 = Maxpool(in_channels=128, out_channels=256, maxpool_size=(2, 2))
        self.maxpool3 = Maxpool(in_channels=256, out_channels=512, maxpool_size=(2, 2))
        self.maxpool4 = Maxpool(in_channels=512, out_channels=1024, maxpool_size=(2, 2))
        self.upsampling1 = Upsampling(in_channels=1024,out_channels=512,upsampling_size=(2,2))
        self.upsampling2 = Upsampling(in_channels=512, out_channels=256, upsampling_size=(2, 2))
        self.upsampling3 = Upsampling(in_channels=256, out_channels=128, upsampling_size=(2, 2))
        self.upsampling4 = Upsampling(in_channels=128, out_channels=64, upsampling_size=(2, 2))
        self.resconcate1 = res_concate(in_channels=64,num_res=1)
        self.resconcate2 = res_concate(in_channels=128, num_res=2)
        self.resconcate3 = res_concate(in_channels=256, num_res=3)
        self.resconcate4 = res_concate(in_channels=512, num_res=4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.dropout= nn.Dropout(p = 0.6)
    def forward(self,x):
        x = self.fisrt_conv(x)
        x = self.resmaxpool(x)
        x1 = self.resupsampling(x)
        x1_drop = self.dropout(x1)
        x2 = self.maxpool1(x1_drop)
        x3 = self.maxpool2(x2)
        x4 = self.maxpool3(x3)
        x = self.maxpool4(x4)
        x = self.dropout(x)
        x = self.upsampling1(x,self.resconcate4(x4))
        x = self.upsampling2(x,self.resconcate3(x3))
        x = self.upsampling3(x,self.resconcate2(x2))
        x = self.upsampling4(x,self.resconcate1(x1))
        x = self.dropout(x)
        x = self.final_conv(x)
        return x
if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(5, 3, 2048, 12).to(device)
    model = Res_U_Res_Net().to(device)
    output = model(x)
    # summary(model, input_size=(3, 2048, 12))
    print(output.shape)