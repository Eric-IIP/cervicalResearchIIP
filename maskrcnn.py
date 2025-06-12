import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_maskrcnn_model(num_classes, pretrained=True):
    """Create Mask R-CNN model"""
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model
    
    
def dice_coefficient(pred_mask, gt_mask):
        intersection = (pred_mask & gt_mask).float().sum()
        union = pred_mask.float().sum() + gt_mask.float().sum()
        if union == 0:
            return torch.tensor(1.0 if intersection == 0 else 0.0)
        return (2. * intersection) / union
    
    
class Trainer2MaskRCNN:
    """
    Modified Trainer2 class specifically for Mask R-CNN
    """
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.DataLoader = None,
                 validation_DataLoader: torch.utils.data.DataLoader = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 ):
        
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        
        # Move model to device
        self.model.to(device)
    def run_trainer(self):
        self.train(self)
        self.validate(self)
    
    def train(self):
        device = self.device
        model = self.model
        epochs = self.epochs
        optimizer = self.optimizer
        training_data = self.training_DataLoader
        
        for epoch in range(epochs):
            model.train()
            for x, y in training_data:
                x = list(img.to(device) for img in x)
                y = [{k: v.to(device) for k, v in t.items()} for t in y]
                
                loss_dict = model(x, y)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            print(f"Epoch {epoch}, loss: {losses.item():.4f}")
        
    
    def validate(self):
        validation_data = self.validation_DataLoader
        device = self.device
        model = self.model
        model.eval()
        dice_scores = []
        
        with torch.no_grad():
            for images, targets in validation_data:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for pred, target in zip(outputs, targets):
                    pred_masks = pred['masks'] > 0.5  # shape: [N, 1, H, W]
                    gt_masks = target['masks'] > 0.5  # shape: [N_gt, H, W]

                    # Flatten pred_masks to [N, H, W]
                    pred_masks = pred_masks.squeeze(1)

                    if len(pred_masks) == 0 or len(gt_masks) == 0:
                        dice_scores.append(torch.tensor(0.0))
                        continue

                    # Naive matching: compute Dice between each pred and each GT, take max for each GT
                    for gt_mask in gt_masks:
                        best_dice = 0
                        for pred_mask in pred_masks:
                            dice = dice_coefficient(pred_mask, gt_mask)
                            best_dice = max(best_dice, dice)
                        dice_scores.append(best_dice)

        mean_dice = torch.stack(dice_scores).mean().item()
        print(f"Validation Dice: {mean_dice:.4f}")