# U-Net: 使用Pytorch进行语义分割
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/milesial/PyTorch-UNet/main.yml?logo=github&style=for-the-badge)](#)
[![Docker Image](https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge)](https://hub.docker.com/r/milesial/unet)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-v1.13+-red.svg?logo=PyTorch&style=for-the-badge)](https://pytorch.org/)
[![Python Version](https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge)](#)

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)

这是针对 Kaggle 的 [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) 从高清晰度图像中进行分割的 [U-Net](https://arxiv.org/abs/1505.04597) 的定制实现。

- [快速开始](#快速开始)
  - [不使用 Docker](#不使用-docker)
  - [使用 Docker](#使用-docker)
- [描述](#描述)
- [用法](#用法)
  - [Docker](#docker)
  - [训练](#训练)
  - [预测](#预测)
- [权重和偏差](#权重和偏差)
- [预训练模型](#预训练模型)
- [数据](#数据)

## 快速开始

### 不使用 Docker

1. [安装 CUDA](https://developer.nvidia.com/cuda-downloads) 

2. [安装 PyTorch 1.13 或更高版本](https://pytorch.org/get-started/locally/) 

3. 克隆仓库
```bash
git clone https://github.com/lsewcx/Pytorch-UNet.git
```

4. 安装依赖
```bash
pip install -r requirements.txt
```

4. 下载数据并开始训练：
```bash
unzip Pytorch-UNet/ceping/NEU_Seg-main.zip
```

### 使用 Docker

1. [安装 Docker 19.03 或更高版本：](https://docs.docker.com/get-docker/) 
```bash
curl https://get.docker.com  | sh && sudo systemctl --now enable docker
```
2. [安装 NVIDIA 容器工具包：](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey  | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
3. [下载并运行镜像：](https://hub.docker.com/repository/docker/milesial/unet) 
```bash
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet
```

4. 下载数据并开始训练：
```bash
cd Pytorch-UNet
unzip ceping/NEU_Seg-main.zip
python train.py --classes 4 --batch-size 16 --epochs 50 --scale 0.5
```

5. 推理
```bash 
python predict.py --model best_model.pth
```

## 描述
这个模型是从 5000 张图像从头开始训练的，并且在超过 100000 张测试图像上得分为 [Dice 系数](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) 0.988423。

它可以轻松用于多类分割、肖像分割、医学分割等。

## 用法
**注意：使用 Python 3.6 或更新版本**

### Docker

在 [DockerHub](https://hub.docker.com/repository/docker/milesial/unet) 上提供了一个包含代码和依赖项的 docker 镜像。
你可以使用 ([docker >=19.03](https://docs.docker.com/get-docker/)) 下载并进入容器：

```console
docker run -it --rm --shm-size=8g --ulimit memlock=-1 --gpus all milesial/unet
```


### 训练

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

默认情况下，`scale` 设置为 0.5，因此如果你希望获得更好的结果（但使用更多内存），请将其设置为 1。

自动混合精度也可用 `--amp` 标志。[混合精度](https://arxiv.org/abs/1710.03740) 允许模型使用更少的内存，并在最近的 GPU 上通过使用 FP16 算术更快。建议启用 AMP。

### 预测

训练模型并将其保存为 `MODEL.pth` 后，你可以很容易地通过 CLI 在你的图像上测试输出掩码。

要预测单个图像并保存它：

`python predict.py -i image.jpg -o output.jpg`

要预测多个图像并显示它们而不保存它们：

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel white
  --scale SCALE, -s SCALE
                        Scale factor for the input images
```

你可以使用 `--model MODEL.pth` 指定使用哪个模型文件。

## 权重和偏差

使用 [Weights & Biases](https://wandb.ai/) 可以实时可视化训练进度。损失曲线、验证曲线、权重和梯度直方图以及预测掩码都记录在该平台上。

启动训练时，控制台会打印一个链接。点击它进入你的仪表板。如果你有现有的 W&B 账户，你可以通过设置 `WANDB_API_KEY` 环境变量来链接它。如果没有，它将创建一个匿名运行，该运行在 7 天后自动删除。

## 预训练模型
Carvana 数据集的 [预训练模型](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) 可用。也可以从 torch.hub 加载：

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
```
可用的比例为 0.5 和 1.0。

## 数据
Carvana 数据可在 [Kaggle 网站](https://www.kaggle.com/c/carvana-image-masking-challenge/data) 上获得。

你也可以使用辅助脚本来下载它：

```
bash scripts/download_data.sh
```

输入图像和目标掩码应分别位于 `data/imgs` 和 `data/masks` 文件夹中（请注意，由于数据加载器的贪婪性，`imgs` 和 `masks` 文件夹不应包含任何子文件夹或其他文件）。对于 Carvana，图像是 RGB 格式，掩码是黑白的。

只要你确保它在 `utils/data_loading.py` 中正确加载，你也可以使用你自己的数据集。
