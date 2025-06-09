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

class ExponentialLogCE_DiceLoss(nn.Module):
    def __init__(self, num_class, alpha = 1.0, smoothness = 1.0, beta_ce = 1.0, gamma_dice = 1.0):
        """ Custom exponential and cross entropy exponential logarithmic loss

        Args:
            num_class (_type_): number of classes to segment
            alpha (float, optional): coefficient for exponential in log. Defaults to 1.0.
            smoothness (float, optional): smoothing coefficient for dice calculation. Defaults to 1.0.
            ce_weight (_type_, optional): weight tensors of cross entropy (optional). Defaults to None.
            beta_ce (float, optional): coefficient for cross entropy loss. Defaults to 1.0.
            gamma_dice (float, optional): coefficient for dice loss. Defaults to 1.0.
        """
        super(ExponentialLogCE_DiceLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.smoothness = smoothness
        self.beta_ce = beta_ce
        self.gamma_dice = gamma_dice
        self.CEloss = nn.CrossEntropyLoss()
        
    def forward(self, preds, targets):
        """

        Args:
            preds (_type_): Predicted tensor of the model, shape of: [B, C, H, W] batches, channels, height, width
            targets (_type_): Ground truth labels, the answer y, shape of: [B, H, W] batches, height, width
        Returns:
            combined dice and cross entropy loss processed through exponential logarithmic operation
        """
        
        # cross entropy loss
        ce = self.CEloss(preds, targets)
        
        #predictions to softmax
        probabilities = F.softmax(preds, dim = 1)
        
        #One hot encode targets
        targets_onehot = F.one_hot(targets, self.num_class).permute(0, 3, 1, 2).float()
        
        #Dice loss per class
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * targets_onehot, dims)
        cardinality = torch.sum(probabilities + targets_onehot, dims)
        dice_loss_mean = (1.0 -  (2.0 * intersection + self.smoothness) / (cardinality + self.smoothness)).mean()
        
        #exponential log 
        exp_log_ce = torch.log1p_(torch.exp(-self.alpha * ce)).pow(self.beta_ce)
        exp_log_dice = torch.log1p(torch.exp(-self.alpha * dice_loss_mean)).pow(self.gamma_dice)
        #exp_log_dice_mean = exp_log_dice.mean()
        
        expDiceCEloss= exp_log_ce + exp_log_dice
        
        return expDiceCEloss
        
