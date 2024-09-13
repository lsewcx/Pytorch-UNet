import torch
from unet.unet_model import UNetInception  # 确保导入路径正确

# 创建模型实例
model = UNetInception(n_channels=3, n_classes=4, bilinear=False)

# 创建一个示例输入张量
x = torch.randn(1, 3, 256, 256)  # 假设输入图像大小为 256x256，通道数为 3

# 将模型转换为 ONNX 格式并保存
torch.onnx.export(model, x, "unet_inception_model.onnx", opset_version=11)