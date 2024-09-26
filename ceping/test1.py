import glob
import numpy as np

def check_shapes(directory: str):
    # 获取所有 .npy 文件
    files = sorted(glob.glob(directory + '/*.npy'))
    
    for file in files:
        data = np.load(file)
        if data.shape == (199, 199):
            print(f"File {file} has shape {data.shape}")
        else:
            print(f"File {file} has shape {data.shape}")

if __name__ == "__main__":
    gt_dir = r'Pytorch-UNet\ceping\test_predictions'
    check_shapes(gt_dir)