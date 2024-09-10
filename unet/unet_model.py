""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models

'''
在原本的unet基础上减少了层数512->1024的部分删除了
'''
# class UNet_Attention(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.5):
#         super(UNet_Attention, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 32)
#         self.down1 = Down(32, 64)
#         self.down2 = Down(64, 128)
#         self.down3 = Down(128, 256)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(256, 512 // factor)
#         self.up1 = Up(512, 256 // factor, bilinear)
#         self.up2 = Up(256, 128 // factor, bilinear)
#         self.up3 = Up(128, 64 // factor, bilinear)
#         self.up4 = Up(64, 32, bilinear)
#         self.outc = OutConv(32, n_classes)
        
#         # 添加 Dropout 层
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x5 = self.dropout(x5)  # 在下采样的最后一层添加 Dropout
#         x = self.up1(x5, x4)
#         x = self.dropout(x)  # 在上采样层之间添加 Dropout
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

class UNet_Attention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.5):
        super(UNet_Attention, self).__init__()
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
        
        # 添加残差连接的卷积层
        self.res1 = nn.Conv2d(n_channels, 32, kernel_size=1, padding=0, stride=1)
        self.res2 = nn.Conv2d(32, 64, kernel_size=1, padding=0, stride=1)
        self.res3 = nn.Conv2d(64, 128, kernel_size=1, padding=0, stride=1)
        self.res4 = nn.Conv2d(128, 256, kernel_size=1, padding=0, stride=1)
        self.res5 = nn.Conv2d(256, 512 // factor, kernel_size=1, padding=0, stride=1)
        
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