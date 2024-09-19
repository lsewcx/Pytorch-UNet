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
from tqdm import tqdm
import json
import sys
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

def calculate_iou(pred, gt, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def seg(pred_dir, gt_dir, num_classes):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

    assert len(pred_files) == len(gt_files), "Prediction and GT files count do not match"

    all_ious = np.zeros((len(pred_files), num_classes))

    for i, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
        pred = np.load(os.path.join(pred_dir, pred_file))
        gt = np.load(os.path.join(gt_dir, gt_file))

        ious = calculate_iou(pred, gt, num_classes)
        all_ious[i] = ious

    mean_ious = np.mean(all_ious, axis=0)

    for cls, iou in enumerate(mean_ious):
        print(f"Class {cls} Mean IoU: {iou}")
    return mean_ious

def calculate_miou(all_ious, classes_to_include):
    """
    计算mean intersection over union
    :param all_ious: 所有类别的IoU列表
    :param classes_to_include: 要包含的类别索引列表
    :return: miou
    """
    selected_ious = [all_ious[cls] for cls in classes_to_include]
    return np.mean(selected_ious)


if __name__ == "__main__":
    # 训练好的模型的路径
    model_path = 'best_model.pth'

    score = 0
    ###计算模型参数分数###
    # total_params = count_model_parameters(model_path)
    # norm_params = total_params / 1_000_000
    # print(f"模型的参数总量为: {norm_params} M.")
    # score_para = 0
    # if norm_params > 17:
    #     score_para = 10
    # else:
    #     if norm_params < 1:
    #         score_para = 70
    #     else:
    #         score_para = 70 - 15 / 4 * (norm_params - 1)
    # print(f"模型参数的分数为{score_para}")
    # score += score_para
    ###################

    ####计算class IoU分数####
    pred_dir = 'test_predictions/'
    base_dir = 'baseline_predictions/'
    gt_dir = 'test_ground_truths/'
    num_classes = 4  # 总的分类数
    classes_to_include = [1, 2, 3]  # 只包含分类 1, 2, 3
    improvement_threshold = 0.06
    pre_IoU = seg(pred_dir, gt_dir, num_classes)
    base_IoU = seg(base_dir, gt_dir, num_classes)
    unet_miou = calculate_miou(base_IoU, classes_to_include)
    mymodel_miou = calculate_miou(pre_IoU, classes_to_include)
    thr = math.floor(math.sqrt(100 - 40) / improvement_threshold)
    # thr = 130
    for pre, base in zip(pre_IoU, base_IoU):
        delta = pre - base
        if delta >= improvement_threshold:
            score_class = 100
        else:
            if delta <= 0:
                score_class = 0
            else:
                score_class = 40 + (thr * delta) ** 2
        print(f"分数：{score_class}")
        score += score_class
    print(f"最终分数：{score}")
    results = {
        "UNet": {
            "Class1_IoU": base_IoU[1],
            "Class2_IoU": base_IoU[2],
            "Class3_IoU": base_IoU[3],
            "mIoU": unet_miou,
            "FPS": 0,
            "Parameters": 0
        },
        # "OursModel": {
        #     "Class1_IoU": pre_IoU[1],
        #     "Class2_IoU": pre_IoU[2],
        #     "Class3_IoU": pre_IoU[3],
        #     "mIoU": mymodel_miou,
        #     "FPS": 0,
        #     "Parameters": norm_params
        # }
    }

    json_str = json.dumps(results, indent=4)

    # 将 JSON 字符串保存到 TXT 文件
    with open('results.txt', 'w') as f:
        f.write(json_str)
    with open('results.json', 'w') as f:
        f.write(json_str)