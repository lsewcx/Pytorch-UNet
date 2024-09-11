""" Full assembly of the parts to form the complete network """

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
        x = self.dropout(x)  # 在上采样层之间添加 Dropout
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
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
    def __init__(self, n_channels, n_classes, bilinear=False, dropout_rate=0.4):
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
    """
    The Attention-UNet implementation based on PyTorch.
    As mentioned in the original paper, author proposes a novel attention gate (AG)
    that automatically learns to focus on target structures of varying shapes and sizes.
    Models trained with AGs implicitly learn to suppress irrelevant regions in an input image while
    highlighting salient features useful for a specific task.

    The original article refers to
    Oktay, O, et, al. "Attention u-net: Learning where to look for the pancreas."
    (https://arxiv.org/pdf/1804.03999.pdf).

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,in_channels=3,  num_classes=4, pretrained=None):
        super(UNetAttention, self).__init__()
        self.encoder = Encoder(in_channels, [64, 128, 256, 512])
        filters = [64, 128, 256, 512, 1024]
        self.up5 = UpConv(ch_in=filters[4], ch_out=filters[3])
        self.att5 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_out=filters[2])
        self.up_conv5 = ConvBlock(ch_in=filters[4], ch_out=filters[3])

        self.up4 = UpConv(ch_in=filters[3], ch_out=filters[2])
        self.att4 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_out=filters[1])
        self.up_conv4 = ConvBlock(ch_in=filters[3], ch_out=filters[2])

        self.up3 = UpConv(ch_in=filters[2], ch_out=filters[1])
        self.att3 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_out=filters[0])
        self.up_conv3 = ConvBlock(ch_in=filters[2], ch_out=filters[1])

        self.up2 = UpConv(ch_in=filters[1], ch_out=filters[0])
        self.att2 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_out=filters[0] // 2)
        self.up_conv2 = ConvBlock(ch_in=filters[1], ch_out=filters[0])

        self.conv_1x1 = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x5, (x1, x2, x3, x4) = self.encoder(x)
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat([x4, d5], dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)

        logit = self.conv_1x1(d2)
        logit_list = [logit]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            self.load_state_dict(torch.load(self.pretrained))


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_out):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_out)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_out)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_out, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        res = x * psi
        return res


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    def __init__(self, input_channels, filters):
        super(Encoder, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        down_channels = filters
        self.down_sample_list = nn.ModuleList([
            self.down_sampling(channel, channel * 2)
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.ReLU(inplace=True))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)