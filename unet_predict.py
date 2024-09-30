import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
from unet import *
import json
from utils.utils import plot_img_and_mask
import time  # 导入time模块

def predict_img(net, imgs, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    imgs = [torch.from_numpy(BasicDataset.preprocess(None, img, scale_factor, is_mask=False)) for img in imgs]
    imgs = torch.stack(imgs)
    imgs = imgs.to(device=device, dtype=torch.float32)
        
    with torch.no_grad():    
        output = net(imgs).cpu()
        output = F.interpolate(output, (200, 200), mode='bilinear')  # 调整输出大小为200x200
        n_classes = 4
        if n_classes > 1:
            masks = output.argmax(dim=1)
        else:
            masks = torch.sigmoid(output) > out_threshold
    return masks.long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--output-dir', '-o', metavar='OUTPUT_DIR', default='test_predictions', help='Directory to save output images')
    parser.add_argument('--input-dir', '-i', metavar='INPUT_DIR', required=True, help='Directory of input images')  # 添加输入路径参数
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--model-name', '-model_name', type=str, default='UNet_less', help='Model name')
    parser.add_argument('--batch-size', '-b', type=int, default=1, help='Batch size for inference')  # 添加 batch size 参数

    return parser.parse_args()

def get_output_filenames(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    return args.output_dir

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_dir = args.input_dir
    out_dir = get_output_filenames(args)

    if args.model_name == 'UNet_More_Less':
        net = UNet_More_Less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == 'UNet_less':
        net = UNet_less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == 'UNetInception':
        net = UNetInception(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == 'UNetAttention':
        net = UNetAttention(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == 'UNet_plusplus':
        net = UNetPlusPlus(n_channels=3, n_classes=args.classes,use_deconv=True, align_corners=False, is_ds=True)
    elif args.model_name == 'UNetPlusPlusInception':
        net = UNetPlusPlusInception(n_channels=3, n_classes=args.classes, use_deconv=True, align_corners=False, is_ds=True)
    elif args.model_name == 'UNet':
        try:
            import segmentation_models_pytorch as smp
            net = smp.Unet(
            encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=4,                      # model output channels (number of classes in your dataset)
            )
            total_params = sum(p.numel() for p in net.parameters())
            logging.info(f"模型的参数量: {total_params / 1e6:.2f}M")

        except ImportError:
            pass
    else:
        raise ValueError(f'Unknown model name: {args.model_name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net = torch.load(args.model)

    logging.info('Model loaded!')

    total_inference_time = 0  # 累积推理时间
    num_images = 0  # 处理的图像数量

    image_files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    batch_size = args.batch_size

    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        imgs = [Image.open(os.path.join(in_dir, f)) for f in batch_files]

        logging.info(f'Predicting batch {i // batch_size + 1} ...')
        print(f'Predicting batch {i // batch_size + 1} ...')  # 添加 print 语句
        
        start_time = time.time()  # 记录开始时间
        masks = predict_img(net=net,
                            imgs=imgs,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            device=device)
        end_time = time.time()  # 记录结束时间

        inference_time = end_time - start_time
        total_inference_time += inference_time  # 累积推理时间
        num_images += len(batch_files)  # 增加图像计数

        fps = len(batch_files) / inference_time
        logging.info(f'Inference time: {inference_time:.4f} seconds, FPS: {fps:.2f}')
        print(f'Inference time: {inference_time:.4f} seconds, FPS: {fps:.2f}')  # 添加 print 语句

        for j, mask in enumerate(masks):
            output_filename = os.path.join(out_dir, f'prediction_{batch_files[j][:-4]}.npy')
            np.save(output_filename, mask)
            logging.info(f'Mask saved to {output_filename}')
            print(f'Mask saved to {output_filename}')  # 添加 print 语句

    if num_images > 0:
        avg_fps = num_images / total_inference_time  # 计算平均FPS
        logging.info(f'Average FPS: {avg_fps:.2f}')
        print(f'Average FPS: {avg_fps:.2f}')  # 添加 print 语句
    else:
        logging.info('No images processed.')
        print('No images processed.')  # 添加 print 语句