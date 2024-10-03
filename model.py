from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
InPlaceABN = None


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Decoder: UNet
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # conv2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, up_channels, concat_channels, out_channels):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(
            in_channels, up_channels, kernel_size=2, stride=2
        )

        self.conv = DoubleConv(concat_channels, out_channels)

    def forward(self, x, encoder_x):
        x = self.up_conv(x)

        x = torch.cat([encoder_x, x], dim=1)

        x = self.conv(x)

        return x


# # full model: ResNet34_UNet
# class self_net(nn.Module):
#     def __init__(self, n_classes=4):
#         super().__init__()
#         self.__name__ = "ResNet34_UNet"

#         # encoder: ResNet34
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(inplace=True)

#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = nn.Sequential(
#             BasicBlock(64, 64),
#             BasicBlock(64, 64),
#             BasicBlock(64, 64),
#         )

#         self.layer2 = nn.Sequential(
#             BasicBlock(64, 128, stride=2, downsample=Downsample(64, 128, stride=2)),
#             BasicBlock(128, 128),
#             BasicBlock(128, 128),
#             BasicBlock(128, 128),
#         )

#         self.layer3 = nn.Sequential(
#             BasicBlock(128, 256, stride=2, downsample=Downsample(128, 256, stride=2)),
#             BasicBlock(256, 256),
#             BasicBlock(256, 256),
#             BasicBlock(256, 256),
#             BasicBlock(256, 256),
#             BasicBlock(256, 256),
#         )

#         self.layer4 = nn.Sequential(
#             BasicBlock(256, 512, stride=2, downsample=Downsample(256, 512, stride=2)),
#             BasicBlock(512, 512),
#             BasicBlock(512, 512),
#         )

#         # concat layer3
#         self.up1 = Up(
#             in_channels=512,
#             up_channels=256,
#             concat_channels=256 + 256,
#             out_channels=256,
#         )

#         # concat layer2
#         self.up2 = Up(
#             in_channels=256,
#             up_channels=128,
#             concat_channels=128 + 128,
#             out_channels=128,
#         )

#         # concat maxpool_layer1
#         self.up3 = Up(
#             in_channels=128,
#             up_channels=128,
#             concat_channels=128 + 64,
#             out_channels=64,
#         )

#         # concat conv1_bn_1_relu1
#         self.up4 = Up(
#             in_channels=64,
#             up_channels=64,
#             concat_channels=64 + 64,
#             out_channels=64,
#         )

#         self.up5 = nn.ConvTranspose2d(
#             in_channels=64, out_channels=32, kernel_size=2, stride=2
#         )

#         self.out_conv = nn.Conv2d(in_channels=32, out_channels=n_classes, kernel_size=1)

#     def forward(self, x):
#         # encoder
#         block1 = self.conv1(x)
#         block1 = self.bn1(block1)
#         block1 = self.relu1(block1)

#         block2 = self.maxpool1(block1)
#         block2 = self.layer1(block2)

#         block3 = self.layer2(block2)

#         block4 = self.layer3(block3)

#         block5 = self.layer4(block4)

#         # decoder
#         x = self.up1(block5, block4)
#         x = self.up2(x, block3)
#         x = self.up3(x, block2)
#         x = self.up4(x, block1)
#         x = self.up5(x)

#         out = self.out_conv(x)

#         return out

class self_net(nn.Module):
    def __init__(self):
        super(self_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 4, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x1 = torch.relu(self.bn1(self.conv1(x)))
        x2 = torch.relu(self.bn2(self.conv2(x1)))
        x3 = torch.relu(self.bn3(self.conv3(x2)))
        x4 = self.deconv1(x3)
        x5 = self.deconv2(x4 + x2)
        return x5







