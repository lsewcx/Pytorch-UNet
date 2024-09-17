import json
from PIL import Image
import numpy as np

# 读取JSON文件
with open('layer_outputs.json', 'r') as json_file:
    layer_outputs = json.load(json_file)

# 遍历每一层的输出
for layer_name, layer_output in layer_outputs.items():
    # 将列表形式的输出转换为NumPy数组
    layer_output = np.array(layer_output)
    
    # 将NumPy数组转换为PIL图像
    layer_image = Image.fromarray((layer_output * 255).astype(np.uint8))
    
    # 显示图像
    layer_image.show()
