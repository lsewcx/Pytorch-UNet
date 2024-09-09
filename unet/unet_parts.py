""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: 输入通道维度
            WH: 输入特征图的空间维度
            M: 分支的数量
            G: 卷积组的数量
            r: 计算d，向量s的压缩倍数，C/r
            stride: 步长，默认为1
            L: 矢量z的最小维度，默认为32
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        # 使用不同kernel size的卷积，增加不同的感受野
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # 全局平均池化
        self.gap = nn.AvgPool2d(int(WH / stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        # 全连接层
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        ''' Split操作'''
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
 
        ''' Fuse操作'''
        fea_U = torch.sum(feas, dim=1)
        fea_s = self.gap(fea_U).squeeze_()
        fea_z = self.fc(fea_s)
 
        ''' Select操作'''
        for i, fc in enumerate(self.fcs):
            # fc-->d*c维
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        # 计算attention权重
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        # 最后一步，各特征图与对应的注意力权重相乘，得到输出特征图V
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert in_channels % num_heads == 0, "in_channels should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0)
        self.attention = nn.Softmax(dim=-1)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.size()
        qkv = self.qkv(x).reshape(B, 3 * self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.chunk(3, dim=1)
        q = q * self.scale
        attn = self.attention(q @ k.transpose(-2, -1))
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        return self.out(x)

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
            nn.ReLU(inplace=True),
            MultiHeadAttention(out_channels, 8)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
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
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
