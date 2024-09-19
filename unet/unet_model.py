""" Full assembly of the parts to form the complete network """

import json
from .unet_parts import *
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models

'''
在原本的unet基础上减少了层数512->1024的部分删除了
到现在为止效果最好的模型
'''
class UNet_less(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.5):
        super(UNet_less, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        
        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)  # 在下采样的最后一层添加 Dropout
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.dropout(x)  # 在上采样层之间添加 Dropout
        logits = self.outc(x)


        return logits
    
class UNet_More_Less(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.5):
        super(UNet_More_Less, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)  # 原来是32
        self.down1 = Down(16, 32)  # 原来是32, 64
        self.down2 = Down(32, 64)  # 原来是64, 128
        self.down3 = Down(64, 128)  # 原来是128, 256
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)  # 原来是256, 512 // factor
        self.up1 = Up(256, 128 // factor, bilinear)  # 原来是512, 256 // factor
        self.up2 = Up(128, 64 // factor, bilinear)  # 原来是256, 128 // factor
        self.up3 = Up(64, 32 // factor, bilinear)  # 原来是128, 64 // factor
        self.up4 = Up(32, 16, bilinear)  # 原来是64, 32
        self.outc = OutConv(16, n_classes)  # 原来是32

        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)  # 在下采样的最后一层添加 Dropout
        x = self.up1(x5, x4)
        x = self.dropout(x)  # 在上采样层之间添加 Dropout
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    
'''
把参数量缩的更小了但是添加了inception v2模块
'''
class UNetInception(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.6):
        super(UNetInception, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvInceptionResNetV2(n_channels, 8)  # 原来是16
        self.down1 = Down(8, 16)  # 原来是16, 32
        self.down2 = Down(16, 32)  # 原来是32, 64
        self.down3 = Down(32, 64)  # 原来是64, 128
        factor = 2 if bilinear else 1
        self.down4 = Down(64, 128 // factor)  # 原来是128, 256 // factor
        self.up1 = Up(128, 64 // factor, bilinear)  # 原来是256, 128 // factor
        self.up2 = Up(64, 32 // factor, bilinear)  # 原来是128, 64 // factor
        self.up3 = Up(32, 16 // factor, bilinear)  # 原来是64, 32 // factor
        self.up4 = Up(16, 8, bilinear)  # 原来是32, 16
        self.outc = OutConv(8, n_classes)  # 原来是16

        # 添加残差连接的卷积层
        self.res1 = nn.Conv2d(8, 8, kernel_size=1, padding=0, stride=1)  # 原来是16
        self.res2 = nn.Conv2d(16, 16, kernel_size=1, padding=0, stride=1)  # 原来是32
        self.res3 = nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1)  # 原来是64
        self.res4 = nn.Conv2d(64, 64 // factor, kernel_size=1, padding=0, stride=1)  # 原来是128
        self.res5 = nn.Conv2d(128, 128 // factor, kernel_size=1, padding=0, stride=1)  # 原来是256

        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # 下采样部分
        x1 = self.inc(x)
        x1_res = self.res1(x1)  # 残差连接
        x2 = self.down1(x1 + x1_res)
        
        x2_res = self.res2(x2)  # 残差连接
        x3 = self.down2(x2 + x2_res)
        
        x3_res = self.res3(x3)  # 残差连接
        x4 = self.down3(x3 + x3_res)
        
        x4_res = self.res4(x4)  # 残差连接
        x5 = self.dropout(self.down4(x4 + x4_res))  # 在下采样的最后一层添加 Dropout
        
        # 上采样部分
        x = self.up1(x5, x4)
        x = self.dropout(x)  # 在上采样层之间添加 Dropout
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetAttention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.5):
        super(UNetAttention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)  # 减少通道数
        self.down1 = Down(16, 32)  # 减少通道数
        self.down2 = Down(32, 64)  # 减少通道数
        self.down3 = Down(64, 128)  # 减少通道数
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)  # 减少通道数
        self.up1 = Up(256, 128 // factor, bilinear)  # 减少通道数
        self.att1 = AttentionBlock(F_g=128 // factor, F_l=128, F_out=64 // factor)  # 减少通道数
        self.up2 = Up(128, 64 // factor, bilinear)  # 减少通道数
        self.att2 = AttentionBlock(F_g=64 // factor, F_l=64, F_out=32 // factor)  # 减少通道数
        self.up3 = Up(64, 32 // factor, bilinear)  # 减少通道数
        self.att3 = AttentionBlock(F_g=32 // factor, F_l=32, F_out=16 // factor)  # 减少通道数
        self.up4 = Up(32, 16, bilinear)  # 减少通道数
        self.outc = OutConv(16, n_classes)  # 减少通道数
        
        # 添加 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)  # 在下采样的最后一层添加 Dropout
        x = self.up1(x5, x4)
        x = self.att1(g=x, x=x4)  # 添加注意力机制
        x = self.dropout(x)  # 在上采样层之间添加 Dropout
        x = self.up2(x, x3)
        x = self.att2(g=x, x=x3)  # 添加注意力机制
        x = self.up3(x, x2)
        x = self.att3(g=x, x=x2)  # 添加注意力机制
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


