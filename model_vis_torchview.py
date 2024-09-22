import torch
from torchview import draw_graph
from unet import *  # 确保导入路径正确

# 创建模型实例
model = self_net(n_channels=3, n_classes=4)

# 创建一个示例输入张量
x = torch.randn(1, 3, 224, 224)  # 假设输入图像大小为 256x256，通道数为 3

# 生成计算图
graph = draw_graph(model, input_size=(1, 3, 224, 224))

# 保存计算图为 PDF 文件
# graph.visual_graph.render('unet_inception_model_torchview', format='pdf')
graph.visual_graph.render('unet_inception_model_torchview', format='png')