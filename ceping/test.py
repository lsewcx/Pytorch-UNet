import numpy as np
from torchvision.transforms.functional import to_tensor, normalize
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm


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
    """(convolution => [BN] => ReLU) * 2 => MultiHeadAttention"""

    def __init__(self, in_channels, out_channels, num_heads=8, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MultiHeadAttention(out_channels, num_heads)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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
            self.conv = DoubleConv(in_channels, out_channels, num_heads=8)  # 将 num_heads 设置为 8
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, num_heads=8)  # 将 num_heads 设置为 8


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/pytorch/pytorch/issues/22656
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 128)
        self.up1 = Up(256, 64, bilinear)
        self.up2 = Up(128, 32, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class NEUDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = [img for img in os.listdir(images_dir) if img.endswith('.jpg')]
        print(f"Found {len(self.images)} images in {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        mask_name = os.path.join(self.masks_dir, self.images[idx].replace('.jpg', '.png'))
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")
        mask = np.array(mask)

        # 将255视为背景，将其转换为0
        mask[mask == 255] = 0

        # 确保所有值都在0到3范围内
        mask = np.clip(mask, 0, 3)

        mask_one_hot = np.eye(4)[mask]  # 一共四个分类，包括背景
        mask_one_hot = mask_one_hot.transpose(2, 0, 1)  # 转换为CHW格式

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask_one_hot = self.mask_transform(mask_one_hot)

        return image, mask_one_hot

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # 添加一个批处理维度
    return image.to(device)  # 将图像移动到设备上

def save_predictions_as_npy(model, test_images_dir, output_dir, device):
    test_images = [img for img in os.listdir(test_images_dir) if img.endswith('.jpg')]
    for image_name in tqdm(test_images):  # 使用tqdm包装test_images
        image_path = os.path.join(test_images_dir, image_name)
        image = load_and_preprocess_image(image_path, device)
        with torch.no_grad():  # 确保在推理模式下不计算梯度
            output = model(image)
            predictions = torch.argmax(output, dim=1)  # 获取最大概率的索引
            predictions = predictions.squeeze().cpu().numpy()  # 转换为numpy数组
            output_path = os.path.join(output_dir, "prediction_" + image_name.replace('.jpg', '.npy'))
            np.save(output_path, predictions)  # 保存预测结果

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检测设备
    model = torch.load('model.pth')
    model = model.to(device)  # 将模型移动到设备上
    model.eval()  # 设置模型为评估模式
    test_images_dir = r"NEU_Seg-main\images\test"
    output_dir = "test_predictions"
    os.makedirs(output_dir, exist_ok=True)
    save_predictions_as_npy(model, test_images_dir, output_dir, device)