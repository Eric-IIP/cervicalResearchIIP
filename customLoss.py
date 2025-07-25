import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
class EnsembleInspiredLoss(nn.Module):
    def __init__(self, dst_threshold=2.0, upp_dst_threshold=8.0, 
                 area_ratio_threshold=200, image_size=256):
        super(EnsembleInspiredLoss, self).__init__()
        
        self.dst_threshold = dst_threshold
        self.upp_dst_threshold = upp_dst_threshold
        self.area_ratio_threshold = area_ratio_threshold
        self.image_size = image_size
        
        # Loss component weights
        self.distance_weight = 0.5
        self.connectivity_weight = 0.5
        self.area_consistency_weight = 0.5
        self.base_ce_weight = 1.0
        
        # Base loss
        self.ce_loss = nn.CrossEntropyLoss()
        
        # We'll create coordinate grids dynamically to ensure correct device
        
    def compute_centroid(self, mask):
        """
        Compute centroid of a soft mask (differentiable version of mean position)
        mask: [B, H, W] soft mask values between 0 and 1
        """
        device = mask.device
        B, H, W = mask.shape
        
        # Create coordinate grids on the same device as mask
        y_coords = torch.arange(H, device=device, dtype=mask.dtype).view(1, H, 1)
        x_coords = torch.arange(W, device=device, dtype=mask.dtype).view(1, 1, W)
        
        # Weighted average of coordinates
        total_mass = mask.sum(dim=(1, 2), keepdim=True) + 1e-8  # [B, 1, 1]
        
        # Compute weighted centroids
        y_centroid = (mask * y_coords).sum(dim=(1, 2), keepdim=True) / total_mass  # [B, 1, 1]
        x_centroid = (mask * x_coords).sum(dim=(1, 2), keepdim=True) / total_mass  # [B, 1, 1]
        
        return y_centroid.squeeze(), x_centroid.squeeze()  # [B], [B]
    
    def euclidean_distance_loss(self, pred_soft, target_onehot):
        """
        Penalize when predicted and ground truth centroids are far apart
        Similar to your euclidean_dst function but differentiable
        """
        num_classes = pred_soft.shape[1]
        distance_losses = []
        device = pred_soft.device
        
        for c in range(num_classes):
            pred_mask = pred_soft[:, c]  # [B, H, W]
            target_mask = target_onehot[:, c]  # [B, H, W]
            
            # Skip if no target pixels for this class
            class_present = target_mask.sum(dim=(1, 2)) > 0  # [B]
            
            if class_present.sum() > 0:
                # Compute centroids
                pred_y, pred_x = self.compute_centroid(pred_mask)
                target_y, target_x = self.compute_centroid(target_mask)
                
                # Euclidean distance
                distance = torch.sqrt((pred_x - target_x)**2 + (pred_y - target_y)**2 + 1e-8)
                
                # Penalty based on your thresholds
                penalty = torch.where(
                    distance < self.dst_threshold,
                    torch.zeros_like(distance),  # Good case - no penalty
                    torch.where(
                        distance < self.upp_dst_threshold,
                        distance / self.upp_dst_threshold,  # Moderate penalty
                        torch.ones_like(distance) * 2.0  # High penalty for far distances
                    )
                )
                
                # Only apply to batches where class is present
                penalty = penalty * class_present.float()
                distance_losses.append(penalty.mean())
        
        if distance_losses:
            return torch.stack(distance_losses).mean()
        else:
            return torch.tensor(0.0, device=device, dtype=pred_soft.dtype)
    
    def connectivity_loss(self, pred_soft):
        """
        Encourage connectivity by penalizing fragmented predictions
        Soft approximation of your connected() function
        """
        connectivity_losses = []
        device = pred_soft.device
        
        for c in range(pred_soft.shape[1]):
            class_pred = pred_soft[:, c]  # [B, H, W]
            
            # Compute "connectivity" using local coherence
            # Penalize when neighboring pixels have very different predictions
            
            # Horizontal differences
            h_diff = torch.abs(class_pred[:, :, 1:] - class_pred[:, :, :-1])
            # Vertical differences  
            v_diff = torch.abs(class_pred[:, 1:, :] - class_pred[:, :-1, :])
            
            # Average local variation (higher = more fragmented)
            h_variation = h_diff.mean(dim=(1, 2))  # [B]
            v_variation = v_diff.mean(dim=(1, 2))  # [B]
            
            connectivity_penalty = (h_variation + v_variation) / 2
            connectivity_losses.append(connectivity_penalty.mean())
        
        if connectivity_losses:
            return torch.stack(connectivity_losses).mean()
        else:
            return torch.tensor(0.0, device=device, dtype=pred_soft.dtype)
    
    def area_consistency_loss(self, pred_soft, target_onehot):
        """
        Penalize large differences in predicted vs target area
        Similar to your SA_u vs SA_m comparison
        """
        area_losses = []
        device = pred_soft.device
        
        for c in range(pred_soft.shape[1]):
            pred_area = pred_soft[:, c].sum(dim=(1, 2))  # [B]
            target_area = target_onehot[:, c].sum(dim=(1, 2))  # [B]
            
            # Skip if no target area
            class_present = target_area > 0
            
            if class_present.sum() > 0:
                # Relative area difference
                area_ratio = pred_area / (target_area + 1e-8)
                
                # Penalty for ratios far from 1.0
                area_penalty = torch.abs(torch.log(area_ratio + 1e-8))
                
                # Apply threshold similar to your area_ratio_threshold
                normalized_penalty = torch.clamp(area_penalty / 2.0, 0, 1)  # Normalize
                
                # Only apply to batches where class is present
                penalty = normalized_penalty * class_present.float()
                area_losses.append(penalty.mean())
        
        if area_losses:
            return torch.stack(area_losses).mean()
        else:
            return torch.tensor(0.0, device=device, dtype=pred_soft.dtype)
    
    def missing_class_penalty(self, pred_soft, target_onehot):
        """
        Heavy penalty when model completely misses a class that should be present
        Similar to your logic gate 1 and 2 cases
        """
        missing_penalties = []
        device = pred_soft.device
        
        for c in range(pred_soft.shape[1]):
            pred_area = pred_soft[:, c].sum(dim=(1, 2))  # [B]
            target_area = target_onehot[:, c].sum(dim=(1, 2))  # [B]
            
            # Class should be present but prediction is almost zero
            should_exist = target_area > 10  # Threshold for "class exists"
            barely_predicted = pred_area < 1.0  # Almost no prediction
            
            missing_mask = should_exist & barely_predicted
            penalty = missing_mask.float() * 5.0  # Heavy penalty
            
            missing_penalties.append(penalty.mean())
        
        if missing_penalties:
            return torch.stack(missing_penalties).mean()
        else:
            return torch.tensor(0.0, device=device, dtype=pred_soft.dtype)
    
    def forward(self, pred, target):
        """
        pred: [B, C, H, W] - model predictions (logits)
        target: [B, H, W] - ground truth class indices
        """
        # Convert to soft predictions and one-hot targets
        pred_soft = torch.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), pred.shape[1]).float()
        target_onehot = target_onehot.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Base classification loss
        ce_loss = self.ce_loss(pred, target)
        
        # Custom loss components based on your ensemble logic
        distance_loss = self.euclidean_distance_loss(pred_soft, target_onehot)
        connectivity_loss = self.connectivity_loss(pred_soft)
        #area_loss = self.area_consistency_loss(pred_soft, target_onehot)
        #missing_loss = self.missing_class_penalty(pred_soft, target_onehot)
        
        # Combine all components
        total_loss = (
            self.base_ce_weight * ce_loss +
            self.distance_weight * distance_loss +
            self.connectivity_weight * connectivity_loss #+
            #self.area_consistency_weight * area_loss +
            #2.0 * missing_loss  # Heavy weight for missing classes
        )
        
        # Store components for monitoring
        self.loss_components = {
            'ce': ce_loss.item(),
            'distance': distance_loss.item(),
            'connectivity': connectivity_loss.item(),
            #'area': area_loss.item(),
            #'missing': missing_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss
    
    def get_loss_breakdown(self):
        """Get detailed breakdown of loss components"""
        if hasattr(self, 'loss_components'):
            return self.loss_components
        return {}