import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('Pytorch-UNet/ceping/NEU_Seg-main/annotations/training/000201.png')
img1=mpimg.imread('Pytorch-UNet/ceping/NEU_Seg-main/annotations/training/004355.png')
img2 = mpimg.imread('Pytorch-UNet/ceping/NEU_Seg-main/annotations/training/003898.png')

img_1=mpimg.imread('Pytorch-UNet/ceping/NEU_Seg-main/images/training/000201.jpg')
img1_1=mpimg.imread('Pytorch-UNet/ceping/NEU_Seg-main/images/training/004355.jpg')
img2_1=mpimg.imread('Pytorch-UNet/ceping/NEU_Seg-main/images/training/003898.jpg')

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(img)
axs[0, 0].set_title('Annotation 1')
axs[0, 0].axis('off')

axs[0, 1].imshow(img1)
axs[0, 1].set_title('Annotation 2')
axs[0, 1].axis('off')

axs[0, 2].imshow(img2)
axs[0, 2].set_title('Annotation 3')
axs[0, 2].axis('off')

axs[1, 0].imshow(img_1)
axs[1, 0].set_title('Image 1')
axs[1, 0].axis('off')

axs[1, 1].imshow(img1_1)
axs[1, 1].set_title('Image 2')
axs[1, 1].axis('off')

axs[1, 2].imshow(img2_1)
axs[1, 2].set_title('Image 3')
axs[1, 2].axis('off')

plt.show()