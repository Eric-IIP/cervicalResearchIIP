import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: Tensor of shape [B, C, H, W] (raw model outputs)
        targets: Tensor of shape [B, H, W] (ground truth class indices)
        """
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        probs = F.softmax(logits, dim=1)  # convert logits to probabilities

        dims = (0, 2, 3)  # dimensions for sum: batch, height, width
        intersection = torch.sum(probs * targets_one_hot, dims)
        union = torch.sum(probs + targets_one_hot, dims)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()  # mean over classes

        return dice_loss

class DiceCELoss(nn.Module):
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W] (class indices)
        """
        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets)
        return self.dice_weight * loss_dice + self.ce_weight * loss_ce
