import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_plusplus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_plusplus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        self.up1_0 = Up(512, 256 // factor, bilinear)
        self.up2_0 = Up(256, 128 // factor, bilinear)
        self.up3_0 = Up(128, 64 // factor, bilinear)
        self.up4_0 = Up(64, 32, bilinear)

        self.up1_1 = Up(256 + 256 // factor, 256 // factor, bilinear)
        self.up2_1 = Up(128 + 128 // factor, 128 // factor, bilinear)
        self.up3_1 = Up(64 + 64 // factor, 64 // factor, bilinear)

        self.up1_2 = Up(256 + 256 // factor + 256 // factor, 256 // factor, bilinear)
        self.up2_2 = Up(128 + 128 // factor + 128 // factor, 128 // factor, bilinear)

        self.up1_3 = Up(256 + 256 // factor + 256 // factor + 256 // factor, 256 // factor, bilinear)

        self.outc = OutConv(256 // factor, n_classes)

    def forward(self, x):
        x0_0 = self.inc(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x3_1 = self.up1_0(x4_0, x3_0)
        x2_1 = self.up2_0(x3_1, x2_0)
        x1_1 = self.up3_0(x2_1, x1_0)
        x0_1 = self.up4_0(x1_1, x0_0)

        x2_2 = self.up1_1(x3_1, x2_1)
        x1_2 = self.up2_1(x2_2, x1_1)
        x0_2 = self.up3_1(x1_2, x0_1)

        x1_3 = self.up1_2(x2_2, x1_2)
        x0_3 = self.up2_2(x1_3, x0_2)

        x0_4 = self.up1_3(x1_3, x0_3)

        logits = self.outc(x0_4)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
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
