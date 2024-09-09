import os
from PIL import Image
import numpy as np

'''
因为数据集没有标注文件，所有的标注信息都是通过像素值来表示的，所以我们需要找到所有图片中的唯一非零像素值。
输出{1,2,3}
'''

def unique_nonzero_pixel_values_in_images_with_pil(path):
    unique_values = set()  # 创建一个空集合，用于存储唯一的非零像素值
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".png"):  # 确保只处理PNG文件
                img_path = os.path.join(root, filename)
                img = Image.open(img_path)
                img_array = np.array(img)  # 将图片转换为numpy数组
                nonzero_values = img_array[img_array != 0]  # 找到所有非零值
                unique_values.update(np.unique(nonzero_values))  # 更新集合中的唯一非零像素值

    print(f"Total unique nonzero pixel values: {len(unique_values)}")
    for item in unique_values:  
        print(item)
    return unique_values

unique_nonzero_pixel_values = unique_nonzero_pixel_values_in_images_with_pil("开发文件/NEU_Seg-main")
