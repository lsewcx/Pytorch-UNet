import argparse
import logging
import os
import random
import sys
from networkx import turan_graph
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from model import self_net
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from model import self_net

torch.autograd.set_detect_anomaly(True)

dir_img = Path("./NEU_Seg-main/images/training")
dir_mask = Path("./NEU_Seg-main/annotations/training")
test_img = Path("./NEU_Seg-main/images/test")
test_mask = Path("./NEU_Seg-main/annotations/test")
dir_checkpoint = Path("./checkpoints/")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # 将目标张量转换为one-hot编码格式
        targets = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    # 将模型移动到指定设备
    model.to(device)
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # # 2. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(
    #     dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    # )
    n_train = len(dataset)
    try:
        test_dataset = CarvanaDataset(test_img, test_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        test_dataset = BasicDataset(test_img, test_mask, img_scale)
    n_val = len(test_dataset)
    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True
    )
    train_loader = DataLoader(dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            img_scale=img_scale,
            amp=amp,
        )
    )

    logger.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    """
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5
    )  # goal: maximize Dice score
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = FocalLoss()
    global_step = 0
    best_val_score = float("-inf")

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(
                    device.type if device.type != "mps" else "cpu", enabled=amp
                ):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        F.softmax(masks_pred, dim=1).float(),
                        F.one_hot(true_masks, num_classes=masks_pred.shape[1])
                        .permute(0, 3, 1, 2)
                        .float(),
                        multiclass=True,
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                )
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                division_step = n_train // (5 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace("/", ".")
                            if (
                                value.grad is not None
                                and (
                                    torch.isinf(value.grad) | torch.isnan(value.grad)
                                ).any()
                            ):
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms["Weights/" + tag] = wandb.Histogram(
                                        value.data.cpu()
                                    )
                                if not (
                                    torch.isinf(value.grad) | torch.isnan(value.grad)
                                ).any():
                                    histograms["Gradients/" + tag] = wandb.Histogram(
                                        value.grad.data.cpu()
                                    )

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logger.info("Validation Dice score: {}".format(val_score))
                        if val_score > best_val_score:
                            best_val_score = val_score
                            logger.info(
                                f"\nNew best model with Dice score: {val_score}"
                            )
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            torch.save(model, f"{val_score}.pth")
                            logger.info("Best model saved!")
                        try:
                            experiment.log(
                                {
                                    "learning rate": optimizer.param_groups[0]["lr"],
                                    "validation Dice": val_score,
                                    "images": wandb.Image(images[0].cpu()),
                                    "masks": {
                                        "true": wandb.Image(
                                            true_masks[0].float().cpu()
                                        ),
                                        "pred": wandb.Image(
                                            masks_pred.argmax(dim=1)[0].float().cpu()
                                        ),
                                    },
                                    "step": global_step,
                                    "epoch": epoch,
                                    **histograms,
                                }
                            )
                        except:
                            pass

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                torch.save(model, "model.pth")
                logger.info(f"Checkpoint {epoch} saved!")


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=50, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=1, help="Downscaling factor of the images"
    )
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )
    parser.add_argument(
        "--classes", "-c", type=int, default=4, help="Number of classes"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="UNet_less", help="Model name"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f'Using device {torch.device("cuda" if torch.cuda.is_available() else "cpu")}'
    )
    """
    UNet_less效果最好到现在为止
    """
    # if args.model == 'UNet_More_Less':
    #     model = UNet_More_Less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model == 'UNet_less':
    #     model = UNet_less(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model == 'UNetInception':
    #     model = UNetInception(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model == 'UNetAttention':
    #     model = UNetAttention(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # elif args.model == 'UNet_plusplus':
    #     model = UNetPlusPlus(n_channels=3, n_classes=args.classes,use_deconv=True, align_corners=False, is_ds=True)
    # elif args.model == 'UNetPlusPlusInception':
    #     model = UNetPlusPlusInception(n_classes=args.classes, n_channels=3, use_deconv=True, align_corners=False, is_ds=True)
    if args.model == "selfnet":
        model = self_net()
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"模型的参数量: {total_params / 1e6:.2f}M")
        # try:
        #     import segmentation_models_pytorch as smp
        #     # model = smp.DeepLabV3Plus(
        #     #     encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     #     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     #     classes=4,                      # model output channels (number of classes in your dataset)
        #     # )
        #     model = smp.Unet(
        #     encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=4,                      # model output channels (number of classes in your dataset)
        #     )
        #     total_params = sum(p.numel() for p in model.parameters())
        #     logging.info(f"模型的参数量: {total_params / 1e6:.2f}M")

        # except ImportError:
        #     pass
    else:
        raise ValueError(f"Unknown model name: {args.model}")
    logger.info(f"Network: {model.__class__.__name__}")
    model = model.to(memory_format=torch.channels_last)

    # logger.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.n_classes} output channels (classes)\n')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logger.info(f"Model loaded from {args.load}")

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )