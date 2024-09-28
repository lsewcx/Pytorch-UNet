import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import transformer
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
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
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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

# class self_net(nn.Module):
#     def __init__(self, n_channels=3, n_classes=4):
#         super(self_net, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = False

#         self.inc = DoubleConv(n_channels, 64)  # 原来是64
#         self.down1 = Down(64, 128)  # 原来是64, 128
#         self.down2 = Down(128,256)  # 原来是128, 256
#         self.down3 = Down(256,512)  # 原来是256, 512
#         factor = 2 if self.bilinear else 1
#         self.down4 = Down(512, 1024 // factor)  # 原来是512, 1024 // factor


#         self.up1 = Up(1024, 512 // factor, self.bilinear)  # 原来是1024, 512 // factor
#         self.up2 = Up(512, 256 // factor, self.bilinear)  # 原来是512, 256 // factor
#         self.up3 = Up(256, 128 // factor, self.bilinear)  # 原来是256, 128 // factor
#         self.up4 = Up(128, 64, self.bilinear)  # 原来是128, 64
#         self.outc = OutConv(64, n_classes)  # 原来是64

#     def forward(self, x):
#         # 下采样部分
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         # 移除了self.ppm(x5)
        
#         # 上采样部分
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

class self_net(nn.Module):
    def __init__(self, n_channels=3, n_classes=4):
        super(self_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Use ResNet50 as the encoder
        self.encoder = resnet50(pretrained=True)

        # Use Transformer as the middle part
        self.transformer = transformer.Transformer()

        # Use U-Net as the decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder part
        x1 = self.encoder(x)

        # Transformer part
        x2 = self.transformer(x1)

        # Decoder part
        x = self.up1(x2, x1)
        x = self.up2(x, x1)
        x = self.up3(x, x1)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits