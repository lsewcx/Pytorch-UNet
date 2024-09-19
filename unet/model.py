import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvInceptionResNetV2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvInceptionResNetV2(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvInceptionResNetV2(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvInceptionResNetV2(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class self_net(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, bilinear=False):
        super(self_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConvInceptionResNetV2(n_channels, 32)  # 原来是16
        self.down1 = Down(32, 64)  # 原来是16, 32
        self.down2 = Down(64, 128)  # 原来是32, 64
        self.down3 = Down(128, 256)  # 原来是64, 128
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)  # 原来是128, 256 // factor
        self.up1 = Up(512, 256 // factor, bilinear)  # 原来是256, 128 // factor
        self.up2 = Up(256, 128 // factor, bilinear)  # 原来是128, 64 // factor
        self.up3 = Up(128, 64 // factor, bilinear)  # 原来是64, 32 // factor
        self.up4 = Up(64, 32, bilinear)  # 原来是32, 16
        self.outc = OutConv(32, n_classes)  # 原来是16

        # 添加残差连接的卷积层
        self.res1 = nn.Conv2d(32, 32, kernel_size=1, padding=0, stride=1)  # 原来是16
        self.res2 = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)  # 原来是32
        self.res3 = nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1)  # 原来是64
        self.res4 = nn.Conv2d(256, 256 // factor, kernel_size=1, padding=0, stride=1)  # 原来是128
        self.res5 = nn.Conv2d(512, 512 // factor, kernel_size=1, padding=0, stride=1)  # 原来是256

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
        x5 = self.down4(x4 + x4_res)  # 在下采样的最后一层去掉 Dropout
        
        # 上采样部分
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits