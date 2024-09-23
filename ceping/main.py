import glob
import os
import torch
import math
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import sys
from torchmetrics.segmentation import MeanIoU



sys.path.append('../')

def count_model_parameters(model_path):
    """
    计算模型参数总量
    :param model_path: 模型文件路径（.pt或.pth），要求选手使用 torch.save(model, 'model.pth') 保存模型
    :return: 模型参数总量
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    # 加载模型
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # 计算参数总量
    total_params = sum(p.numel() for p in model.parameters())
    return total_params




def segmiou(num_classes: int, pred_dir: str, truth_dir: str):
    # 初始化 MeanIoU
    miou = MeanIoU(num_classes=num_classes, include_background=False)
    
    # 获取预测和真值文件列表，并确保顺序一致
    target_files = sorted(glob.glob(truth_dir + '/*.npy'))
    pred_files = sorted(glob.glob(pred_dir + '/*.npy'))
    
    # 检查文件数量是否一致
    assert len(target_files) == len(pred_files), "预测文件和真值文件数量不一致"
    
    # 读取数据并确保数据类型和形状一致
    target_data = torch.stack([torch.from_numpy(np.load(file)).long() for file in target_files])
    pred_data = torch.stack([torch.from_numpy(np.load(file)).long() for file in pred_files])
    
    # 检查并调整大小一致
    if target_data.shape != pred_data.shape:
        min_shape = np.minimum(target_data.shape, pred_data.shape)
        target_data = target_data[:, :min_shape[1], :min_shape[2]]
        pred_data = pred_data[:, :min_shape[1], :min_shape[2]]
    
    # 计算 mIoU
    miou_result = miou(pred_data, target_data)
    
    return miou_result

if __name__ == "__main__":
    # 训练好的模型的路径
    # model_path = 'best_model.pth'

    # score = 0
    # try:
    #     ###计算模型参数分数###
    #     total_params = count_model_parameters(model_path)
    #     norm_params = total_params / 1_000_000
    #     print(f"模型的参数总量为: {norm_params} M.")
    #     score_para = 0
    #     if norm_params > 17:
    #         score_para = 10
    #     else:
    #         if norm_params < 1:
    #             score_para = 70
    #         else:
    #             score_para = 70 - 15 / 4 * (norm_params - 1)
    #     print(f"模型参数的分数为{score_para}")
    #     score += score_para
    #     ###################
    # except FileNotFoundError as e:
    #     print(e)
    #     print("模型文件不存在，跳过模型参数计算。")

    ####计算class IoU分数####
    pred_dir = 'test_predictions/'
    base_dir = 'baseline_predictions/'
    gt_dir = 'test_ground_truths/'
    num_classes = 4  # 总的分类数
    classes_to_include = [1, 2, 3]  # 只包含分类 1, 2, 3
    improvement_threshold = 0.06
    thr = math.floor(math.sqrt(100 - 40) / improvement_threshold)
    
    pred_miou=segmiou(num_classes,pred_dir,gt_dir)
    base_miou=segmiou(num_classes,base_dir,gt_dir)
    print(pred_miou)
    print(base_miou)
    # thr = 130
    # for pre, base in zip(pre_IoU, base_IoU):
    #     delta = pre - base
    #     if delta >= improvement_threshold:
    #         score_class = 100
    #     else:
    #         if delta <= 0:
    #             score_class = 0
    #         else:
    #             score_class = 40 + (thr * delta) ** 2
    #     print(f"分数：{score_class}")
    #     score += score_class
    # print(f"最终分数：{score}")
    # results = {
    #     "UNet": {
    #         "Class1_IoU": base_IoU[1],
    #         "Class2_IoU": base_IoU[2],
    #         "Class3_IoU": base_IoU[3],
    #         "mIoU": unet_miou,
    #         "FPS": 26.66,
    #         "Parameters": 31.04
    #     },
    #     "OursModel": {
    #         "Class1_IoU": pre_IoU[1],
    #         "Class2_IoU": pre_IoU[2],
    #         "Class3_IoU": pre_IoU[3],
    #         "mIoU": mymodel_miou,
    #         "FPS": 0,
    #         "Parameters": norm_params
    #     }
    # }

    # json_str = json.dumps(results, indent=4)

    # # 将 JSON 字符串保存到 TXT 文件
    # with open('results.txt', 'w') as f:
    #     f.write(json_str)
    # with open('results.json', 'w') as f:
    #     f.write(json_str)