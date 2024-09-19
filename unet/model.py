import torch
import torch.nn as nn
import torch.nn.functional as F

# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DepthwiseSeparableConv, self).__init__()
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.relu(x)

# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels, bias=False), 
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             DepthwiseSeparableConv(in_channels, mid_channels),
#             DepthwiseSeparableConv(mid_channels, out_channels),
#             SEBlock(out_channels)  # 添加SE模块
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

# class PyramidPoolingModule(nn.Module):
#     def __init__(self, in_channels, pool_sizes):
#         super(PyramidPoolingModule, self).__init__()
#         self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in pool_sizes])
#         self.bottleneck = nn.Conv2d(in_channels * (len(pool_sizes) + 1), in_channels, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)

#     def _make_stage(self, in_channels, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=size)
#         conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
#         return nn.Sequential(prior, conv)

#     def forward(self, x):
#         h, w = x.size(2), x.size(3)
#         pyramids = [x]
#         for stage in self.stages:
#             pyramids.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
#         output = torch.cat(pyramids, dim=1)
#         output = self.bottleneck(output)
#         return self.relu(output)

# class self_net(nn.Module):
#     def __init__(self, n_channels=3, n_classes=4, bilinear=False):
#         super(self_net, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 32)  # 原来是16
#         self.down1 = Down(32, 64)  # 原来是16, 32
#         self.down2 = Down(64, 128)  # 原来是32, 64
#         self.down3 = Down(128, 256)  # 原来是64, 128
#         factor = 2 if bilinear else 1
#         self.down4 = Down(256, 512 // factor)  # 原来是128, 256 // factor
#         self.ppm = PyramidPoolingModule(512 // factor, [1, 2, 3, 6])  # 添加金字塔池化模块
#         self.up1 = Up(512, 256 // factor, bilinear)  # 原来是256, 128 // factor
#         self.up2 = Up(256, 128 // factor, bilinear)  # 原来是128, 64 // factor
#         self.up3 = Up(128, 64 // factor, bilinear)  # 原来是64, 32 // factor
#         self.up4 = Up(64, 32, bilinear)  # 原来是32, 16
#         self.outc = OutConv(32, n_classes)  # 原来是16

#     def forward(self, x):
#         # 下采样部分
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         # 金字塔池化
#         x5 = self.ppm(x5)
        
#         # 上采样部分
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super(DoubleConv, self).__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class UpSampling(nn.Module):
#     def __init__(self, in_channels, out_channels, n_cat, use_deconv=False, align_corners=False):
#         super(UpSampling, self).__init__()
#         if use_deconv:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#         else:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        
#         self.conv = DoubleConv(n_cat * out_channels, out_channels)

#     def forward(self, high_feature, *low_features):
#         x = self.up(high_feature)
#         for i in range(len(low_features)):
#             diffY = low_features[i].size()[2] - x.size()[2]
#             diffX = low_features[i].size()[3] - x.size()[3]
#             x = F.pad(x, [diffX // 2, diffX - diffX // 2,
#                           diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x, *low_features], dim=1)
#         return self.conv(x)

# class UpSamplingInception(nn.Module):
#     def __init__(self, in_channels, out_channels, n_cat, use_deconv=False, align_corners=False):
#         super(UpSamplingInception, self).__init__()
#         if use_deconv:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#         else:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        
#         self.conv = DoubleConvInceptionResNetV2(n_cat * out_channels, out_channels)

#     def forward(self, high_feature, *low_features):
#         x = self.up(high_feature)
#         for i in range(len(low_features)):
#             diffY = low_features[i].size()[2] - x.size()[2]
#             diffX = low_features[i].size()[3] - x.size()[3]
#             x = F.pad(x, [diffX // 2, diffX - diffX // 2,
#                           diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x, *low_features], dim=1)
#         return self.conv(x)

# class InceptionResNetV2Module(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(InceptionResNetV2Module, self).__init__()
#         self.branch1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         self.branch3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.branch3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

#         self.branch5x5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.branch5x5_2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2)

#         self.branch_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#         self.conv = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)

#         branch3x3 = self.branch3x3_1(x)
#         branch3x3 = self.branch3x3_2(branch3x3)

#         branch5x5 = self.branch5x5_1(x)
#         branch5x5 = self.branch5x5_2(branch5x5)

#         branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
#         x = torch.cat(outputs, 1)
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
    
# class DoubleConvInceptionResNetV2(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#             self.double_conv = nn.Sequential(
#                 nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(mid_channels),
#                 nn.ReLU(inplace=True),
#                 InceptionResNetV2Module(mid_channels, out_channels),
#                 # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#             )

#     def forward(self, x):
#         return self.double_conv(x)
    
# class self_net(nn.Module):
#     def __init__(self, n_classes, n_channels=3, use_deconv=True, align_corners=False, is_ds=True,bilinear=False):
#         super(self_net, self).__init__()
#         self.is_ds = is_ds
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         channels = [16,32, 64, 128, 256]
        
#         self.conv0_0 = DoubleConvInceptionResNetV2(n_channels, channels[0])
#         self.conv1_0 = DoubleConvInceptionResNetV2(channels[0], channels[1])
#         self.conv2_0 = DoubleConvInceptionResNetV2(channels[1], channels[2])
#         self.conv3_0 = DoubleConvInceptionResNetV2(channels[2], channels[3])
#         self.conv4_0 = DoubleConvInceptionResNetV2(channels[3], channels[4])

#         self.up_cat0_1 = UpSamplingInception(channels[1], channels[0], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)
#         self.up_cat1_1 = UpSamplingInception(channels[2], channels[1], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)
#         self.up_cat2_1 = UpSamplingInception(channels[3], channels[2], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)
#         self.up_cat3_1 = UpSamplingInception(channels[4], channels[3], n_cat=2, use_deconv=use_deconv, align_corners=align_corners)

#         self.up_cat0_2 = UpSamplingInception(channels[1], channels[0], n_cat=3, use_deconv=use_deconv, align_corners=align_corners)
#         self.up_cat1_2 = UpSamplingInception(channels[2], channels[1], n_cat=3, use_deconv=use_deconv, align_corners=align_corners)
#         self.up_cat2_2 = UpSamplingInception(channels[3], channels[2], n_cat=3, use_deconv=use_deconv, align_corners=align_corners)

#         self.up_cat0_3 = UpSamplingInception(channels[1], channels[0], n_cat=4, use_deconv=use_deconv, align_corners=align_corners)
#         self.up_cat1_3 = UpSamplingInception(channels[2], channels[1], n_cat=4, use_deconv=use_deconv, align_corners=align_corners)

#         self.up_cat0_4 = UpSamplingInception(channels[1], channels[0], n_cat=5, use_deconv=use_deconv, align_corners=align_corners)

#         self.out_1 = nn.Conv2d(channels[0], n_classes, 1)
#         self.out_2 = nn.Conv2d(channels[0], n_classes, 1)
#         self.out_3 = nn.Conv2d(channels[0], n_classes, 1)
#         self.out_4 = nn.Conv2d(channels[0], n_classes, 1)

#     def forward(self, inputs):
#         # 0 down
#         X0_0 = self.conv0_0(inputs)
#         X1_0 = self.conv1_0(F.max_pool2d(X0_0, 2))
#         X2_0 = self.conv2_0(F.max_pool2d(X1_0, 2))
#         X3_0 = self.conv3_0(F.max_pool2d(X2_0, 2))
#         X4_0 = self.conv4_0(F.max_pool2d(X3_0, 2))

#         # 1 up+concat
#         X0_1 = self.up_cat0_1(X1_0, X0_0)
#         X1_1 = self.up_cat1_1(X2_0, X1_0)
#         X2_1 = self.up_cat2_1(X3_0, X2_0)
#         X3_1 = self.up_cat3_1(X4_0, X3_0)

#         # 2 up+concat
#         X0_2 = self.up_cat0_2(X1_1, X0_0, X0_1)
#         X1_2 = self.up_cat1_2(X2_1, X1_0, X1_1)
#         X2_2 = self.up_cat2_2(X3_1, X2_0, X2_1)

#         # 3 up+concat
#         X0_3 = self.up_cat0_3(X1_2, X0_0, X0_1, X0_2)
#         X1_3 = self.up_cat1_3(X2_2, X1_0, X1_1, X1_2)

#         # 4 up+concat
#         X0_4 = self.up_cat0_4(X1_3, X0_0, X0_1, X0_2, X0_3)

#         # out conv1*1
#         out_1 = self.out_1(X0_1)
#         out_2 = self.out_2(X0_2)
#         out_3 = self.out_3(X0_3)
#         out_4 = self.out_4(X0_4)

#         output = (out_1 + out_2 + out_3 + out_4) / 4

#         if self.is_ds:
#             return output
#         else:
#             return out_4



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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn6 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))
        x5 = self.avg_pool(x)
        x5 = self.relu(self.bn5(self.conv5(x5)))
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn6(self.conv6(x)))
        return x

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, size=feature.shape[-2:], mode="nearest")
            last_inner = inner_block(feature) + inner_top_down
            results.insert(0, layer_block(last_inner))
        return results

class self_net(nn.Module):
    def __init__(self, n_classes, n_channels=3, use_deconv=False, align_corners=False, is_ds=True):
        super(self_net, self).__init__()
        self.is_ds = is_ds
        self.n_channels = n_channels
        self.n_classes = n_classes
        channels = [64, 128, 256, 512,1024]
        
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

        self.aspp = ASPP(channels[4], channels[4])
        self.fpn = FPN([channels[0], channels[1], channels[2], channels[3], channels[4]], channels[0])

    def forward(self, inputs):
        # 0 down
        X0_0 = self.conv0_0(inputs)
        X1_0 = self.conv1_0(F.max_pool2d(X0_0, 2))
        X2_0 = self.conv2_0(F.max_pool2d(X1_0, 2))
        X3_0 = self.conv3_0(F.max_pool2d(X2_0, 2))
        X4_0 = self.conv4_0(F.max_pool2d(X3_0, 2))

        # ASPP
        X4_0 = self.aspp(X4_0)

        # FPN
        fpn_features = self.fpn([X0_0, X1_0, X2_0, X3_0, X4_0])
        X0_0, X1_0, X2_0, X3_0, X4_0 = fpn_features

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
            return output
        else:
            return out_4