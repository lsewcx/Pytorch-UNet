import json
from PIL import Image
import numpy as np

# 假设你已经有了一个包含所有层输出的JSON文件
with open('layer_outputs.json', 'r') as json_file:
    layer_outputs = json.load(json_file)

# 遍历每一层的输出
for layer_name, layer_output in layer_outputs.items():
    # 假设每一层的输出是一个二维数组，表示图像的像素值
    # 这里我们将其转换为PIL图像并显示
    image_array = np.array(layer_output)
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(f'{layer_name}.png')
