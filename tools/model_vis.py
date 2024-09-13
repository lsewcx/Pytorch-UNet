import torch
from torchviz import make_dot
from ..unet import UNetInception  # 假设你的模型定义在 unet_model.py 文件中

# 创建模型实例
model = UNetInception(n_channels=3, n_classes=4, bilinear=False)

# 创建一个示例输入张量
x = torch.randn(1, 3, 256, 256)  # 假设输入图像大小为 256x256，通道数为 3

# 获取模型的输出
y = model(x)

# 生成计算图
dot = make_dot(y, params=dict(model.named_parameters()))

# 保存计算图为 PDF 文件
dot.format = 'pdf'
dot.render('unet_inception_model')