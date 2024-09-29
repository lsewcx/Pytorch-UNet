import torch
import segmentation_models_pytorch as smp

model = torch.load('best_model.pth')

print(model)