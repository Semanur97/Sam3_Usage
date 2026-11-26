"""Loss ve metrikler. Hepsi instance başına hesaplanır, sonra ortalanır."""
import numpy as np
import torch
import torch.nn.functional as F


def dice_loss(logits, target, smooth=1.0):
    """logits, target: [n, H, W]. Her instance için ayrı Dice, sonra ortalama."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean()


def bce_dice_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def instance_iou_dice(pred: np.ndarray, gt: np.ndarray):
    """pred, gt: [N, H, W] bool/0-1. Her instance için (IoU, Dice) dizileri döner."""
    if len(gt) == 0:
        return np.zeros(0), np.zeros(0)
    p = pred.reshape(len(pred), -1).astype(bool)
    g = gt.reshape(len(gt), -1).astype(bool)
    inter = (p & g).sum(1)
    union = (p | g).sum(1)
    psum, gsum = p.sum(1), g.sum(1)
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    dice = np.where(psum + gsum > 0, 2 * inter / np.maximum(psum + gsum, 1), 1.0)
    return iou, dice
