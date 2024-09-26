import os
import numpy as np

def load_npy_files(pred_folder_path, gt_folder_path, file_prefix):
    preds_list = []
    targets_list = []
    for pred_file_name in os.listdir(pred_folder_path):
        if pred_file_name.endswith('.npy'):
            file_id = pred_file_name.replace(file_prefix, '').replace('.npy', '')
            gt_file_name = f'ground_truth_{file_id}.npy'
            pred_file_path = os.path.join(pred_folder_path, pred_file_name)
            gt_file_path = os.path.join(gt_folder_path, gt_file_name)
            if os.path.exists(pred_file_path) and os.path.exists(gt_file_path):
                preds = np.load(pred_file_path).astype(np.int64)
                targets = np.load(gt_file_path).astype(np.int64)
                preds_list.append(preds)
                targets_list.append(targets)
            else:
                print(f"文件 {pred_file_name} 或 {gt_file_name} 不存在，跳过。")

    if preds_list and targets_list:
        preds = np.stack(preds_list)
        targets = np.stack(targets_list)
        return preds, targets
    else:
        return None, None

def calculate_iou(preds, targets, num_classes, exclude_background=True):
    ious = []
    start_cls = 1 if exclude_background else 0  # 如果排除背景，从类别1开始，否则从类别0开始
    for cls in range(start_cls, num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            iou = float('nan')  # 如果没有该类别的预测或目标，IoU为NaN
        else:
            iou = intersection / union
        
        ious.append(iou)

    miou = np.nanmean(ious)  # 计算mIoU，排除NaN值
    return miou, ious

def segmiou(pred_folder_path, gt_folder_path, file_prefix, num_classes, exclude_background=True):
    preds, targets = load_npy_files(pred_folder_path, gt_folder_path, file_prefix)

    if preds is None or targets is None:
        print("预测或目标文件加载失败。")
        return float('nan'), []

    miou, ious = calculate_iou(preds, targets, num_classes, exclude_background)
    return miou, ious

if __name__ == "__main__":
    pred_dir = 'test_predictions/'
    base_dir = 'baseline_predictions/'
    gt_dir = 'test_ground_truths/'
    num_classes = 4  # 总的分类数
    exclude_background = True  # 是否排除背景类别

    pred_miou, pred_per_class_iou = segmiou(pred_dir, gt_dir, 'prediction_', num_classes, exclude_background)
    base_miou, base_per_class_iou = segmiou(base_dir, gt_dir, 'prediction_', num_classes, exclude_background)
    
    for i, iou in enumerate(pred_per_class_iou, start=1 if exclude_background else 0):
        print(f"预测 Class {i} IoU: {iou}")
    print(f"预测 mIoU: {pred_miou}")
    for i, iou in enumerate(base_per_class_iou, start=1 if exclude_background else 0):
        print(f"基线 Class {i} IoU: {iou}")
    print(f"基线 mIoU: {base_miou}")