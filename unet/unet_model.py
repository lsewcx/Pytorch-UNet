""" Full assembly of the parts to form the complete network """

from .unet_parts import *


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


class UNet_Attention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.sk = SKConv(64, 480, 2, 1, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.sk(x)
        logits = self.outc(x)
        return logits