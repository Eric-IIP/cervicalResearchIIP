import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


## dice loss
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

# custom loss function combining Dice and CE
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

## Custom loss inspired by ensemble criteria
# based on distance between centroids, connectivity, area consistency and missing whole class penalty        
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
        #self.area_consistency_weight = 0.5
        self.base_ce_weight =  1.0
        
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
            #self.base_ce_weight * ce_loss +
            distance_loss + connectivity_loss #+
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
    
## Custom loss for class confusion with penalty matrix
class ConfusionPenaltyLoss(nn.Module):
    def __init__(self, num_classes=11, penalty_matrix=None, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
        if penalty_matrix is None:
            penalty_matrix = torch.randn(num_classes, num_classes)
        
        penalty_matrix.fill_diagonal_(1.0) # Correct predictions have a weight of 1
        self.register_buffer("penalty_matrix", penalty_matrix)

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] (raw outputs from network)
        targets: [B, H, W] (ground truth labels, long dtype)
        """
        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)   # [N, C]
        targets_flat = targets.view(-1)                           # [N]
        
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [N]

        with torch.no_grad():
            pred_flat = logits_flat.argmax(dim=1)  # [N]


        
        penalties = self.penalty_matrix.to(targets_flat.device)[targets_flat, pred_flat]  # [N]


        loss = penalties * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.view(B, H, W)
        

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    It addresses class imbalance by down-weighting easy examples, 
    focusing the training on hard-to-classify pixels.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor, optional): A manual rescaling weight given to each class.
                                            If given, has to be a Tensor of size C.
                                            Defaults to None.
            gamma (float, optional): The focusing parameter. Higher values give more 
                                     focus to hard examples. Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model output of shape [B, C, H, W].
            targets (torch.Tensor): Ground truth of shape [B, H, W].
        
        Returns:
            torch.Tensor: The calculated focal loss.
        """
        # Calculate the cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Calculate the probability of the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate the focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_boundary_mask(mask, kernel_size=3):
    """
    Creates a boundary mask from a segmentation mask.
    Args:
        mask (torch.Tensor): The segmentation mask of shape [B, H, W].
        kernel_size (int): The size of the kernel for dilation.
    Returns:
        torch.Tensor: The boundary mask of shape [B, 1, H, W].
    """
    # Unsqueeze to add a channel dimension for conv operations
    mask_one_hot = F.one_hot(mask, num_classes=mask.max() + 1).permute(0, 3, 1, 2).float()
    
    # Use max pooling as a simple and effective dilation operation
    dilated = F.max_pool2d(mask_one_hot, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = 1 - F.max_pool2d(1 - mask_one_hot, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    boundary = (dilated - eroded).sum(dim=1, keepdim=True)
    boundary = (boundary > 0).float()
    return boundary

class BoundaryDiceLoss(nn.Module):
    """
    Dice Loss that focuses only on the boundary regions of the segmentation mask.
    """
    def __init__(self, smooth=1e-6):
        super(BoundaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model output logits [B, C, H, W].
            targets (torch.Tensor): Ground truth labels [B, H, W].
        """
        # Get softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get one-hot encoded target
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Get the boundary mask from the ground truth
        boundary_mask = get_boundary_mask(targets)
        
        # Filter probabilities and targets to only include boundaries
        probs_boundary = probs * boundary_mask
        targets_boundary = targets_one_hot * boundary_mask
        
        # Flatten
        probs_boundary = probs_boundary.reshape(probs_boundary.shape[0], -1)
        targets_boundary = targets_boundary.reshape(targets_boundary.shape[0], -1)
        
        # Calculate intersection and union over the boundary
        intersection = (probs_boundary * targets_boundary).sum()
        dice_score = (2. * intersection + self.smooth) / (probs_boundary.sum() + targets_boundary.sum() + self.smooth)
        
        return 1. - dice_score

# Note: A BoundaryIoULoss would be implemented identically, just with the IoU formula:
# iou_score = (intersection + self.smooth) / (probs_boundary.sum() + targets_boundary.sum() - intersection + self.smooth)


class BoundaryIOU(nn.Module):
    """
    Dice Loss that focuses only on the boundary regions of the segmentation mask.
    """
    def __init__(self, smooth=1e-6):
        super(BoundaryIOU, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model output logits [B, C, H, W].
            targets (torch.Tensor): Ground truth labels [B, H, W].
        """
        # Get softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get one-hot encoded target
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Get the boundary mask from the ground truth
        boundary_mask = get_boundary_mask(targets)
        
        # Filter probabilities and targets to only include boundaries
        probs_boundary = probs * boundary_mask
        targets_boundary = targets_one_hot * boundary_mask
        
        # Flatten
        probs_boundary = probs_boundary.reshape(probs_boundary.shape[0], -1)
        targets_boundary = targets_boundary.reshape(targets_boundary.shape[0], -1)
        
        # Calculate intersection and union over the boundary
        intersection = (probs_boundary * targets_boundary).sum()
        iou_score = (intersection + self.smooth) / (probs_boundary.sum() + targets_boundary.sum() - intersection + self.smooth)
        
        return 1. - iou_score

class CombinedLoss(nn.Module):
    """
    Combines Focal Loss and Boundary Dice Loss with learnable weights.
    The individual loss functions are created internally.
    """
    def __init__(self, focal_gamma=2.0, b_dice_smooth=1e-6):
        """
        Args:
            focal_gamma (float): The gamma parameter for the internal Focal Loss.
            b_dice_smooth (float): The smooth parameter for the internal Boundary Dice Loss.
        """
        super(CombinedLoss, self).__init__()
        # Instantiate internal loss functions
        self.boundary_iou_loss = BoundaryIOU(smooth = b_dice_smooth)
        self.boundary_dice_loss = BoundaryDiceLoss(smooth=b_dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.confusion_penalty_loss = ConfusionPenaltyLoss()
        self.ensemble_inspired_loss = EnsembleInspiredLoss()
        self.dice_loss = DiceLoss()
        self.ce = nn.CrossEntropyLoss()        
        
        # Learnable parameters (log variances for numerical stability)
        self.log_vars = nn.Parameter(torch.zeros(7))

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model output logits [B, C, H, W].
            targets (torch.Tensor): Ground truth labels [B, H, W].
        
        Returns:
            torch.Tensor: The final combined, weighted loss.
        """
        # --- Calculate Loss Term 1: Boundary IOU Loss ---
        loss1 = self.boundary_iou_loss(inputs, targets)
        precision1 = 0.5 * torch.exp(-self.log_vars[0])
        term1 = precision1 * loss1 + 0.5 * self.log_vars[0]
        
        
        # --- Calculate Loss Term 2: Boundary Dice Loss ---
        loss2 = self.boundary_dice_loss(inputs, targets)
        precision2 = 0.5 * torch.exp(-self.log_vars[1])
        term2 = precision2 * loss2 + 0.5 * self.log_vars[1]
        
        
        # --- Calculate Loss Term 3: Focal Loss ---
        loss3 = self.focal_loss(inputs, targets)
        precision3 = 0.5 * torch.exp(-self.log_vars[2])
        term3 = precision3 * loss3 + 0.5 * self.log_vars[2]
        
        # --- Calculate Loss Term 4: Confusion Penalty Loss ---
        loss4 = self.confusion_penalty_loss(inputs, targets)
        precision4 = 0.5 * torch.exp(-self.log_vars[3])
        term4 = precision4 * loss4 + 0.5 * self.log_vars[3]
        
        # --- Calculate Loss Term 5: Ensemble inspired custom Loss ---
        loss5 = self.ensemble_inspired_loss(inputs, targets)
        precision5 = 0.5 * torch.exp(-self.log_vars[4])
        term5 = precision5 * loss5 + 0.5 * self.log_vars[4]
        
        # --- Calculate Loss Term 6: Dice Loss ---
        loss6 = self.dice_loss(inputs, targets)
        precision6 = 0.5 * torch.exp(-self.log_vars[5])
        term6 = precision6 * loss6 + 0.5 * self.log_vars[5]
        
        # --- Calculate Loss Term 7: Cross entropy Loss ---
        loss7 = self.ce(inputs, targets)
        precision7 = 0.5 * torch.exp(-self.log_vars[6])
        term7 = precision7 * loss7 + 0.5 * self.log_vars[6]
        
        return term1 + term2 + term3 + term4 + term5 + term6 + term7


class CombinedLossV2(nn.Module):
    """
    Combines seven loss functions with learnable weights derived from a softmax.
    The weights are guaranteed to be positive and sum to 1.
    """
    def __init__(self, focal_gamma=2.0, b_dice_smooth=1e-6):
        """
        Args:
            focal_gamma (float): The gamma parameter for the internal Focal Loss.
            b_dice_smooth (float): The smooth parameter for the internal Boundary Dice Loss.
        """
        super(CombinedLossV2, self).__init__()
        
        # --- 1. Instantiate internal loss functions ---
        self.boundary_iou_loss = BoundaryIOU(smooth = b_dice_smooth)
        self.boundary_dice_loss = BoundaryDiceLoss(smooth=b_dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.confusion_penalty_loss = ConfusionPenaltyLoss()
        self.ensemble_inspired_loss = EnsembleInspiredLoss()
        self.dice_loss = DiceLoss()
        self.ce = nn.CrossEntropyLoss()        
        
        # --- 2. Learnable parameters ---
        # Define 7 learnable logits, one for each loss.
        # Initializing to zeros means all losses start with equal weight (1/7).
        self.loss_logits = nn.Parameter(torch.zeros(7))

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model output logits [B, C, H, W].
            targets (torch.Tensor): Ground truth labels [B, H, W].
        
        Returns:
            torch.Tensor: The final combined, weighted loss.
        """
        
        # --- 1. Calculate all individual losses ---
        loss1 = self.boundary_iou_loss(inputs, targets)
        loss2 = self.boundary_dice_loss(inputs, targets)
        loss3 = self.focal_loss(inputs, targets)
        loss4 = self.confusion_penalty_loss(inputs, targets)
        loss5 = self.ensemble_inspired_loss(inputs, targets)
        loss6 = self.dice_loss(inputs, targets)
        loss7 = self.ce(inputs, targets)
        
        # --- 2. Calculate weights ---
        # Apply softmax to the logits to get weights that sum to 1
        weights = F.softmax(self.loss_logits, dim=0)
        
        # --- 3. Combine losses ---
        # Stack all losses into a single tensor
        all_losses = torch.stack([loss1, loss2, loss3, loss4, 
                                  loss5, loss6, loss7])
        
        # Apply weights
        # total_loss = w[0]*L1 + w[1]*L2 + ...
        total_loss = torch.sum(weights * all_losses)
        
        return total_loss