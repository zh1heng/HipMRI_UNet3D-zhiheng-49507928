import torch

def dice_coef(preds, targets, eps=1e-6):
    preds = (preds > 0.5).float()
    intersection = (preds * targets).sum()
    return (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)

def dice_loss(preds, targets):
    return 1 - dice_coef(preds, targets)
