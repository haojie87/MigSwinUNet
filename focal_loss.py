import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np

"""
focal loss
"""


# 1. Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        """
        Args:
            output: Tensor of shape (B, n_class, H, W)
            target: Tensor of shape (B, H, W)
        """
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()

        target = make_one_hot(target.unique(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)

        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (output_flat * target_flat).sum()
        loss = 1 - (2. * intersection + self.smooth) / (output_flat.sum() + target_flat.sum() + self.smooth)

        return loss


# Focal Loss
class Focal_Loss(nn.Module):
    """
    Compute focal loss based on cross-entropy loss.

    Example:
        loss = Focal_Loss()
        input = torch.randn(2, 3, 5, 5, requires_grad=True)         # (B, C, H, W)
        target = torch.empty(2, 5, 5, dtype=torch.long).random_(3)  # (B, H, W)
        output = loss(input, target)
        print(output)

    Args:
        alpha: Weight for each class.
    """
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

        self.CE_loss = nn.CrossEntropyLoss(
            reduction='none',
            ignore_index=ignore_index,
            weight=alpha
        )

    def forward(self, output, target):
        """
        Args:
            output: Tensor of shape (B, C, H, W)
            target: Tensor of shape (B, 1, H, W) or (B, H, W)
        """
        target = torch.squeeze(target, dim=1)  # (B, 1, H, W) -> (B, H, W)
        logpt = self.CE_loss(output, target)   # Step 1: compute cross-entropy term
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt  # Step 2: apply focal weighting

        if self.size_average:
            return loss.mean()
        return loss.sum()


class Focal_Loss_z(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss_z, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        eps = 1e-7
        labels = make_one_hot(labels, 6)
        preds = F.softmax(preds, dim=1)
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # (B, C, H, W) -> (B, C, H*W)
        target = labels.view(y_pred.size())  # (B, C, H, W) -> (B, C, H*W)
        ce = -1. * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.nansum(floss, dim=1)
        floss = torch.nanmean(floss)
        return floss
