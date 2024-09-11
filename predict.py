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
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--output-dir', '-o', metavar='OUTPUT_DIR', default='test_predictions', help='Directory to save output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--model-name', '-model_name', type=str, default='UNet_less', help='Model name')

    return parser.parse_args()

def get_output_filenames(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    return args.output_dir

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_dir = "/kaggle/input/neuseg/NEU_Seg-main/images/test"
    out_dir = get_output_filenames(args)

    if args.model_name == 'UNet_More_Less':
        net = UNet_More_Less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == 'UNet_less':
        net = UNet_less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == 'UNetInception':
        net = UNetInception(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model_name == 'UNetAttention':
        net = UNetAttention(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    else:
        raise ValueError(f'Unknown model name: {args.model_name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    # state_dict = torch.load(args.model, map_location=device)
    # net.load_state_dict(state_dict)
    net=torch.load(args.model)

    logging.info('Model loaded!')

    for filename in os.listdir(in_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_img_path = os.path.join(in_dir, filename)
            img = Image.open(full_img_path)

            logging.info(f'Predicting image {filename} ...')
            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            output_filename = os.path.join(out_dir, f'prediction_{filename[:-4]}.npy')
            np.save(output_filename, mask)
            logging.info(f'Mask saved to {output_filename}')
