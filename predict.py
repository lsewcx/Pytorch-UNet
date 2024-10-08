import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset
from model import self_net
import json
from utils.utils import plot_img_and_mask
import time  # 导入time模块

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    n_classes = 4
    with torch.no_grad():    
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    return mask[0].long().squeeze().numpy()

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

    # if args.model_name == 'UNet_More_Less':
    #     net = UNet_More_Less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model_name == 'UNet_less':
    #     net = UNet_less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model_name == 'UNetInception':
    #     net = UNetInception(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model_name == 'UNetAttention':
    #     net = UNetAttention(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model_name == 'UNet_plusplus':
    #     net = UNetPlusPlus(n_channels=3, n_classes=args.classes,use_deconv=True, align_corners=False, is_ds=True)
    # elif args.model_name == 'UNetPlusPlusInception':
    #     net = UNetPlusPlusInception(n_channels=3, n_classes=args.classes, use_deconv=True, align_corners=False, is_ds=True)
    # elif args.model_name == 'UNet':
    #     net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    if args.model_name == 'selfnet':
        net = self_net()
        # import segmentation_models_pytorch as smp
        #     # model = smp.DeepLabV3Plus(
        #     #     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     #     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     #     classes=4,                      # model output channels (number of classes in your dataset)
        #     # )
        # net = smp.Unet(
        # encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        # in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        # classes=4,                      # model output channels (number of classes in your dataset)
        # )
        # net = self_net(n_channels=3, n_classes=args.classes)
    else:
        raise ValueError(f'Unknown model name: {args.model_name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    if args.model_name == 'UNet':
        state_dict = torch.load(args.model, map_location=device)
        mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        total_params = sum(p.numel() for p in net.parameters())
        total_params_m = total_params / 1_000_000  # 转换为百万参数
        logging.info(f'Total parameters: {total_params_m:.2f}M')
    else:
        net = torch.load(args.model)

    logging.info('Model loaded!')

    total_inference_time = 0  # 累积推理时间
    num_images = 0  # 处理的图像数量

    for filename in os.listdir(in_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_img_path = os.path.join(in_dir, filename)
            img = Image.open(full_img_path)

            logging.info(f'Predicting image {filename} ...')
            print(f'Predicting image {filename} ...')  # 添加 print 语句
            
            start_time = time.time()  # 记录开始时间
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)
            end_time = time.time()  # 记录结束时间

            inference_time = end_time - start_time
            total_inference_time += inference_time  # 累积推理时间
            num_images += 1  # 增加图像计数

            fps = 1 / inference_time
            logging.info(f'Inference time: {inference_time:.4f} seconds, FPS: {fps:.2f}')
            print(f'Inference time: {inference_time:.4f} seconds, FPS: {fps:.2f}')  # 添加 print 语句

            output_filename = os.path.join(out_dir, f'prediction_{filename[:-4]}.npy')
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