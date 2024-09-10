import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

def load_images(image_folder, annotation_folder):
    images = []
    annotations = []
    
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                annotation_path = os.path.join(annotation_folder, file.replace('.jpg', '.png'))
                
                if os.path.exists(annotation_path):
                    images.append(image_path)
                    annotations.append(annotation_path)
    
    return images, annotations

def count_pixel_classes(annotation_path):
    with Image.open(annotation_path) as img:
        pixels = list(img.getdata())
        pixel_counts = defaultdict(int)
        for pixel in pixels:
            pixel_counts[pixel] += 1
    return pixel_counts

def show_random_images(images, annotations, num_images=3):
    selected_indices = random.sample(range(len(images)), num_images)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(selected_indices):
        image = Image.open(images[idx])
        annotation = Image.open(annotations[idx])
        
        # 统计标注图中的像素值
        pixel_counts = count_pixel_classes(annotations[idx])
        
        # 上排显示原图
        plt.subplot(2, num_images, i + 1)
        plt.imshow(image)
        plt.title(f'Original Image {idx+1}')
        plt.axis('off')
        
        # 下排显示标注图
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(annotation)
        plt.title(f'Annotation {idx+1}')
        plt.axis('off')
        
        # 在标注图上显示不同像素值的数量
        for pixel_value, count in pixel_counts.items():
            plt.text(10, 10 + 20 * pixel_value, f'Pixel {pixel_value}: {count}', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
    
    # 添加图例
    handles = [plt.Line2D([0], [0], color='black', lw=4, label='Pixel 0: Background'),
               plt.Line2D([0], [0], color='red', lw=4, label='Pixel 1: Class 1'),
               plt.Line2D([0], [0], color='green', lw=4, label='Pixel 2: Class 2'),
               plt.Line2D([0], [0], color='blue', lw=4, label='Pixel 3: Class 3')]
    plt.legend(handles=handles, loc='upper right')
    
    plt.show()

def main():
    # 随机展示三张原始图片和三张标注过后的图片
    image_folder = 'NEU_Seg-main/images/training'  # 替换为你的图片文件夹路径
    annotation_folder = 'NEU_Seg-main/annotations/training'  # 替换为你的标注文件夹路径
    
    # 加载图片和标注
    images, annotations = load_images(image_folder, annotation_folder)
    
    # 展示随机三张图片及其对应的标注
    show_random_images(images, annotations, num_images=3)

if __name__ == "__main__":
    main()