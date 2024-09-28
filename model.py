import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url
from torch import  nn

class ConvBlock(nn.Module):
    """conv-norm-relu"""
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1, norm_layer=None):
        """
        :param in_channels:  输入通道
        :param out_channels: 输出通道
        :param kernel_size: 默认为3
        :param padding:    默认为1，
        :param norm_layer:  默认使用 batch_norm
        """
        super(ConvBlock,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else  nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.convblock(x)

class UNetBlock(nn.Module):
    """conv-norm-relu,conv-norm-relu"""
    def __init__(self, in_channels, out_channels,mid_channels=None,padding=0, norm_layer=None):
        """
        :param in_channels:
        :param out_channels:
        :param mid_channels:  默认 mid_channels==out_channels
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        """
        super(UNetBlock,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.unetblock=nn.Sequential(
            ConvBlock(in_channels,mid_channels,padding=padding,norm_layer=norm_layer),
            ConvBlock(mid_channels, out_channels,padding=padding,norm_layer=norm_layer)
        )
    def forward(self, x):
        return self.unetblock(x)


class UNetUpBlock(nn.Module):
    """Upscaling then unetblock"""

    def __init__(self, in_channels, out_channels,padding=0,norm_layer=None, bilinear=True):
        """
        :param in_channels:
        :param out_channels:
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        :param bilinear:  使用双线性插值，或转置卷积
        """

        super(UNetUpBlock,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels , in_channels // 2,1,1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_channels, out_channels,padding=padding,norm_layer=norm_layer)


    def crop(self,tensor,target_sz):
        _, _, tensor_height, tensor_width = tensor.size()
        diff_y = (tensor_height - target_sz[0]) // 2
        diff_x = (tensor_width - target_sz[1]) // 2
        return tensor[:, :, diff_y:(diff_y + target_sz[0]), diff_x:(diff_x + target_sz[1])]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        x2=self.crop(x2,x1.shape[2:])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class UNetDownBlock(nn.Module):
    """maxpooling-unetblock"""

    def __init__(self, in_channels, out_channels,padding=0, norm_layer=None):
        super(UNetDownBlock,self).__init__()

        self.down=nn.Sequential(
            nn.MaxPool2d(2),
            UNetBlock(in_channels, out_channels,padding=padding, norm_layer=norm_layer),
        )
    def forward(self, inputs):
        return self.down(inputs)


class Unet_Encoder(nn.Module):
    def __init__(self, in_channels,base_channels,level,padding=0,norm_layer=None,):
        super(Unet_Encoder,self).__init__()
        self.encoder=nn.ModuleList()
        for i in range(level):
            if i==0:
                #第一层，特征图尺寸和原图大小一致
                self.encoder.append(UNetBlock(in_channels, base_channels*(2**i),
                                              padding=padding,norm_layer=norm_layer))
            else:
                self.encoder.append(UNetDownBlock( base_channels*(2**(i-1)),  base_channels*(2**i),
                                                   padding=padding,norm_layer=norm_layer))

    def forward(self, inputs):
        features=[]
        for block in self.encoder:
            inputs=block(inputs)
            features.append(inputs)
        return features



class UNet(nn.Module):
    def __init__(self,n_classes,base_channels=64,level=5,padding=0,norm_layer=None,bilinear=True):
        super(UNet, self).__init__()
        self.level=level
        self.base_channels=base_channels
        self.norm_layer=norm_layer
        self.padding=padding
        self.bilinear=bilinear
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()
        self.outBlock=nn.Sequential(nn.Conv2d(base_channels,n_classes,1,1),nn.Sigmoid())
    def build_encoder(self):
        return Unet_Encoder(in_channels=3, base_channels=self.base_channels, level=self.level, padding=self.padding)
    def build_decoder(self):
        decoder=nn.ModuleList()
        for i in range(self.level-1): #有 self.level-1 个上采样块
            in_channels= self.base_channels*(2**(self.level-i-1))
            out_channels= self.base_channels*(2**(self.level-i-2))
            decoder.append(UNetUpBlock(in_channels,out_channels,
                                       padding=self.padding,norm_layer= self.norm_layer,bilinear=self.bilinear))
        return  decoder

    def forward(self,x):
        features=self.encoder(x)[0:self.level]
        # for feat in features:
        #     print(feat.shape)
        assert len(features)==self.level
        x=features[-1]
        for i,up_block in enumerate(self.decoder):
            x=up_block(x,features[-2-i])
            #print("shape:{}".format(x.shape))
        if self.outBlock is not None:
            x=self.outBlock(x)
        #加一个softmax激活函数 或则sigmoid也行
        return  x
    
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, padding=1,bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias)

def conv1x1(in_planes, out_planes, stride=1,bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        if (stride != 1 or inplanes != planes * self.expansion):
            assert  downsample!=None, "downsample can't be None! "
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d # 如果bn层没有自定义，就使用标准的bn层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)  # downsample调整x的维度，F(x)+x一致才能相加
        out += identity
        out = self.relu(out) # 先相加再激活
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        if (stride != 1 or inplanes != planes * self.expansion):
            assert  downsample!=None, "downsample can't be None! "
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion) # 输入的channel数：planes * self.expansion
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class InputStem(nn.Module):
    """
     A  implementation of "ResNet-C " from paper :  "Bag of Tricks for Image Classification with Convolutional Neural Networks"
     replace the 7 × 7 convolution in the input stem with three conservative 3 × 3 convolutions.
    it can be found on the implementations of other models, such as SENet , PSPNet ,DeepLabV3 , and ShuffleNetV2 .
    不同的是，我们这里把步长全部设置为1，获得与输入相同尺寸的特征图，以适应图像分割任务。
    """
    def __init__(self,in_planes,planes,norm_layer=None):
        super(InputStem,self).__init__()
        self.model=nn.Sequential(
            ConvBlock(in_planes,planes,3,1,norm_layer=norm_layer),
            ConvBlock(planes, planes, 3, 1,norm_layer=norm_layer),
            ConvBlock(planes, planes, 3, 1,norm_layer=norm_layer)
        )
    def forward(self, inputs):
        return  self.model(inputs)


class ResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=None,b_RGB=True,base_planes=32):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        inplanes=3 if b_RGB==True else 1
        self.input_stem=InputStem(inplanes,base_planes,norm_layer)
        self.inplanes = base_planes
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  base_planes*2//block.expansion, layers[0])
        self.layer2 = self._make_layer(block,  base_planes*4//block.expansion, layers[1], stride=2)
        self.layer3 = self._make_layer(block,  base_planes*8//block.expansion, layers[2], stride=2)
        self.layer4 = self._make_layer(block,  base_planes*16//block.expansion, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # 生成不同的stage/layer
        # block: block type(basic block/bottle block)
        # blocks: blocks的数量
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
            norm_layer(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer)) # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, blocks): # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def forward(self, x):
        #[ b，c, h，w] c=1 or c=3
        x0 = self.input_stem(x)                 #[b,c1,h, w]
        x1 = self.layer1(self.maxpool(x0))    #[b,c2,h//2, w//2]
        x2 = self.layer2(x1)      #[b,c3,h//4, w//4]
        x3 = self.layer3(x2)   #[b,c4,h//8, w//8]
        x4 = self.layer4(x3)  #[b,c5,h//16, w//16]

        return [x0,x1,x2,x3,x4]

def _resnet(arch, block, layers, pretrained=False, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=True)
        # for key,val in state_dict.items():
        #     print(key)
        model.load_state_dict(state_dict, False)

    return model

"""
1. resnet_net 采用了5个不同尺度的特征图图  level：5
2. 用三个3*3卷积代替 7*7卷积，并且步长全部为1,得到与原始图片尺寸相同的特征
3. base_channels控制着网络的宽度
4.   stride：1   网络输出与输入尺寸相同
"""
class self_net(UNet):
    def __init__(self,n_classes=4,norm_layer=None,bilinear=True,**kwargs):
        self.base_channels = kwargs.get("base_channels",32)  # resnet18 和resnet34 这里为 32 , 64
        level=kwargs.get("level",5)
        self.b_RGB = kwargs.get("level", True)

        padding = 1
        super(self_net,self).__init__(n_classes, self.base_channels,level,padding,norm_layer,bilinear)

    def build_encoder(self):
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2],base_planes= self.base_channels,b_RGB=self.b_RGB )




class Res50_UNet(UNet):
    def __init__(self,n_classes,norm_layer=None,bilinear=True):
        self.base_channels = 64     # resnet50 ，resnet101和resnet152 这里为 64, 128,256
        level = 5
        padding = 1
        super(Res50_UNet,self).__init__(n_classes, self.base_channels,level,padding,norm_layer,bilinear)
    def build_encoder(self):
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3],base_planes=self.base_channels,)

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

