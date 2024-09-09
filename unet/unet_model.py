""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet_Attention(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Attention, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.fpn = FPN([512, 256, 128, 64], 256)  # Adjust channel sizes as needed

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # FPN feature maps
        fpn_features = [x1, x2, x3, x4]

        # Upscaling with FPN
        x = self.up1(x5, x4)
        x = self.fpn([x] + fpn_features[1:])  # Add top feature map
        x = self.up2(x, x3)
        x = self.fpn([x] + fpn_features[2:])  # Add next feature maps
        x = self.up3(x, x2)
        x = self.fpn([x] + fpn_features[3:])  # Add next feature maps
        x = self.up4(x, x1)
        x = self.fpn([x] + fpn_features[4:])  # Add remaining feature maps

        logits = self.outc(x)
        return logits