import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_channel,channel,stride=1,downsample=None):
        super().__init__()

        self.conv1=nn.Conv2d(in_channel,channel,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(channel)

        self.conv2=nn.Conv2d(channel,channel,kernel_size=3,padding=1,bias=False,stride=1, groups=channel)
        self.pointwise = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.bn2=nn.BatchNorm2d(channel)

        self.conv3=nn.Conv2d(channel,channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(channel*self.expansion)

        self.relu=nn.ReLU(False)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.relu(self.bn1(self.conv1(x))) #bs,c,h,w
        out=self.relu(self.bn2(self.pointwise(self.conv2(out)))) #bs,c,h,w
        out=self.relu(self.bn3(self.conv3(out))) #bs,4c,h,w

        if(self.downsample != None):
            residual=self.downsample(residual)

        out+=residual
        return self.relu(out)

    
class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000):
        super().__init__()
        self.in_channel=64
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(False)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True)

        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Linear(512*block.expansion,num_classes)
        self.softmax=nn.Softmax(-1)

    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x))) #bs,112,112,64
        out=self.maxpool(out) #bs,56,56,64

        out=self.layer1(out) #bs,56,56,64*4
        out=self.layer2(out) #bs,28,28,128*4
        out=self.layer3(out) #bs,14,14,256*4
        out=self.layer4(out) #bs,7,7,512*4

        out=self.avgpool(out) #bs,1,1,512*4
        out=out.reshape(out.shape[0],-1) #bs,512*4
        out=self.classifier(out) #bs,1000
        out=self.softmax(out)

        return out

    def _make_layer(self,block,channel,blocks,stride=1):
        downsample = None
        if(stride!=1 or self.in_channel!=channel*block.expansion):
            self.downsample=nn.Conv2d(self.in_channel,channel*block.expansion,stride=stride,kernel_size=1,bias=False)
        layers=[]
        layers.append(block(self.in_channel,channel,downsample=self.downsample,stride=stride))
        self.in_channel=channel*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channel,channel))
        return nn.Sequential(*layers)

def ResNet50(num_classes=1000):
    return ResNet(BottleNeck,[3,4,6,3],num_classes=num_classes)

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=False)

        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            CoordinateAttention(mid_channels, mid_channels),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            CoordinateAttention(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class self_net(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, bilinear=True):
        super(self_net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoder = ResNet50()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = self.encoder.layer1
        self.down2 = self.encoder.layer2
        self.down3 = self.encoder.layer3
        self.down4 = self.encoder.layer4
        self.up1 = Up(2048+1024, 1024, bilinear)
        self.up2 = Up(1024+512, 512, bilinear)
        self.up3 = Up(512+256, 256, bilinear)
        self.up4 = Up(256+64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 200, 200)
    model = UNetResNet50(n_channels=3, n_classes=4)
    output = model(input_tensor)
    print(f'输出形状: {output.shape}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型的参数量: {total_params / 1e6:.2f}M')