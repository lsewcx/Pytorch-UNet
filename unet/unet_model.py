""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torchvision.models as models

def get_resnet50_features():
    resnet50 = models.resnet50(pretrained=True)
    # 移除全连接层，只保留卷积层部分
    return nn.Sequential(*list(resnet50.children())[:-1])

class UNet_Attention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 使用ResNet50作为初始卷积层
        self.inc = get_resnet50_features()
        
        # 根据ResNet50的输出通道数调整Down和Up模块
        self.down1 = Down(2048, 512)  
        self.down2 = Down(512, 256)
        self.down3 = Down(256, 128)
        self.down4 = Down(128, 64)
        
        self.up1 = Up(64, 128, bilinear)
        self.up2 = Up(128, 256, bilinear)
        self.up3 = Up(256, 512, bilinear)
        self.up4 = Up(512, 2048, bilinear)  
        
        self.outc = OutConv(2048, n_classes)

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