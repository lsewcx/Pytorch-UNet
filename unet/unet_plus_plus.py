import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, n_cat, use_deconv=False, align_corners=False):
        super(UpSampling, self).__init__()
        if use_deconv:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        
        self.conv = DoubleConv(n_cat * out_channels, out_channels)

    def forward(self, high_feature, *low_features):
        x = self.up(high_feature)
        for i in range(len(low_features)):
            diffY = low_features[i].size()[2] - x.size()[2]
            diffX = low_features[i].size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, *low_features], dim=1)
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    def __init__(self, n_classes, n_channels=3, use_deconv=False, align_corners=False, is_ds=True):
        super(UNetPlusPlus, self).__init__()
        self.is_ds = is_ds
        self.n_channels = n_channels
        self.n_classes = n_classes
        channels = [32, 64, 128, 256, 512]
        
        self.conv0_0 = DoubleConv(n_channels, channels[0])
        self.conv1_0 = DoubleConv(channels[0], channels[1])
        self.conv2_0 = DoubleConv(channels[1], channels[2])
        self.conv3_0 = DoubleConv(channels[2], channels[3])
        self.conv4_0 = DoubleConv(channels[3], channels[4])

        self.up_cat0_1 = UpSampling(channels[1], channels[0], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)
        self.up_cat1_1 = UpSampling(channels[2], channels[1], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)
        self.up_cat2_1 = UpSampling(channels[3], channels[2], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)
        self.up_cat3_1 = UpSampling(channels[4], channels[3], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)

        self.up_cat0_2 = UpSampling(channels[1], channels[0], n_cat=3, use_deconv=use_deconv, align_corners=align_corners)
        self.up_cat1_2 = UpSampling(channels[2], channels[1], n_cat=3, use_deconv=use_deconv, align_corners=align_corners)
        self.up_cat2_2 = UpSampling(channels[3], channels[2], n_cat=3, use_deconv=use_deconv, align_corners=align_corners)

        self.up_cat0_3 = UpSampling(channels[1], channels[0], n_cat=4, use_deconv=use_deconv, align_corners=align_corners)
        self.up_cat1_3 = UpSampling(channels[2], channels[1], n_cat=4, use_deconv=use_deconv, align_corners=align_corners)

        self.up_cat0_4 = UpSampling(channels[1], channels[0], n_cat=5, use_deconv=use_deconv, align_corners=align_corners)

        self.out_1 = nn.Conv2d(channels[0], n_classes, 1)
        self.out_2 = nn.Conv2d(channels[0], n_classes, 1)
        self.out_3 = nn.Conv2d(channels[0], n_classes, 1)
        self.out_4 = nn.Conv2d(channels[0], n_classes, 1)

    def forward(self, inputs):
        # 0 down
        X0_0 = self.conv0_0(inputs)
        X1_0 = self.conv1_0(F.max_pool2d(X0_0, 2))
        X2_0 = self.conv2_0(F.max_pool2d(X1_0, 2))
        X3_0 = self.conv3_0(F.max_pool2d(X2_0, 2))
        X4_0 = self.conv4_0(F.max_pool2d(X3_0, 2))

        # 1 up+concat
        X0_1 = self.up_cat0_1(X1_0, X0_0)
        X1_1 = self.up_cat1_1(X2_0, X1_0)
        X2_1 = self.up_cat2_1(X3_0, X2_0)
        X3_1 = self.up_cat3_1(X4_0, X3_0)

        # 2 up+concat
        X0_2 = self.up_cat0_2(X1_1, X0_0, X0_1)
        X1_2 = self.up_cat1_2(X2_1, X1_0, X1_1)
        X2_2 = self.up_cat2_2(X3_1, X2_0, X2_1)

        # 3 up+concat
        X0_3 = self.up_cat0_3(X1_2, X0_0, X0_1, X0_2)
        X1_3 = self.up_cat1_3(X2_2, X1_0, X1_1, X1_2)

        # 4 up+concat
        X0_4 = self.up_cat0_4(X1_3, X0_0, X0_1, X0_2, X0_3)

        # out conv1*1
        out_1 = self.out_1(X0_1)
        out_2 = self.out_2(X0_2)
        out_3 = self.out_3(X0_3)
        out_4 = self.out_4(X0_4)

        output = (out_1 + out_2 + out_3 + out_4) / 4

        if self.is_ds:
            return [output]
        else:
            return [out_4]