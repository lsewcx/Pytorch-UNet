import torch
import torch.nn as nn
import torch.nn.functional as F

# 压缩激活模块 (Squeeze-and-Excitation Module)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


# 空洞空间金字塔池化模块 (ASPP)
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=7, dilation=7)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        return out


# 空间注意力模块 (Spatial Attention)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 卷积块注意力模块 (CBAM)
class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.se = SEBlock(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.se(x)
        x = self.sa(x)
        return x


# 完整的 MA-Unet 模型
class self_net(nn.Module):
    def __init__(self):
        super(self_net, self).__init__()
        # 编码器
        self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.encoder_conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        self.aspp = ASPP(1024, 1024)

        # 解码器
        self.decoder_conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.decoder_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.output_conv = nn.Conv2d(64, 4, kernel_size=1)

    def forward(self, x):
        # 编码器
        x1 = F.relu(self.encoder_conv1(x))
        x1 = self.cbam1(x1)

        x2 = F.relu(self.encoder_conv2(x1))
        x2 = self.cbam2(x2)

        x3 = F.relu(self.encoder_conv3(x2))
        x3 = self.cbam3(x3)

        x4 = F.relu(self.encoder_conv4(x3))
        x4 = self.cbam4(x4)

        x5 = F.relu(self.encoder_conv5(x4))
        x5 = self.aspp(x5)

        # 解码器
        x6 = F.relu(self.decoder_conv1(x5))
        x7 = F.relu(self.decoder_conv2(x6))
        x8 = F.relu(self.decoder_conv3(x7))
        x9 = F.relu(self.decoder_conv4(x8))

        out = torch.sigmoid(self.output_conv(x9))

        return out


# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

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

