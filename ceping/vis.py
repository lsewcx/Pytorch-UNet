import numpy as np
import matplotlib.pyplot as plt

# 步骤1: 读取.npy文件
npy_file_path = 'prediction_000330.npy'  # 替换为你的.npy文件路径
data = np.load(npy_file_path)

print(data)