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
    
    model.rpn.nms_thresh = 0.5
    model.rpn.fg_iou_thresh = 0.7
    model.rpn.bg_iou_thresh = 0.3

    model.roi_heads.box_roi_pool.sampling_ratio = 2
    model.roi_heads.mask_roi_pool.sampling_ratio = 2
    
    model.roi_heads.score_thresh = 0.7
    model.roi_heads.nms_thresh = 0.3
    model.roi_heads.detections_per_img = 200
    
        
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
        self.train()
        self.validate()
    
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
                # for k, v in loss_dict.items():
                #     print(f"{k}: {v.item()}")
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            print(f"Epoch {epoch}, loss: {losses.item():.4f}")
        
    
    # def validate(self):
    #     validation_data = self.validation_DataLoader
    #     device = self.device
    #     model = self.model
    #     model.eval()
    #     dice_scores = []
        
    #     with torch.no_grad():
    #         for images, targets in validation_data:
    #             images = list(img.to(device) for img in images)
    #             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                
    #             outputs = model(images)

    #             for pred, target in zip(outputs, targets):
    #                 pred_masks = pred['masks'] > 0.5  # shape: [N, 1, H, W]
    #                 gt_masks = target['masks'] > 0.5  # shape: [N_gt, H, W]

    #                 # Flatten pred_masks to [N, H, W]
    #                 pred_masks = pred_masks.squeeze(1)

    #                 if len(pred_masks) == 0 or len(gt_masks) == 0:
    #                     dice_scores.append(torch.tensor(0.0))
    #                     continue

    #                 # Naive matching: compute Dice between each pred and each GT, take max for each GT
    #                 for gt_mask in gt_masks:
    #                     best_dice = 0
    #                     for pred_mask in pred_masks:
    #                         dice = dice_coefficient(pred_mask, gt_mask)
    #                         best_dice = max(best_dice, dice)
    #                     dice_scores.append(best_dice)

    #     mean_dice = torch.stack(dice_scores).mean().item()
    #     print(f"Validation Dice: {mean_dice:.4f}")
        
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
                    pred_masks = pred['masks'] > 0.5
                    gt_masks = target['masks'] > 0.5

                    pred_masks = pred_masks.squeeze(1)

                    # This part is good, it handles the case where there are no GT masks.
                    if len(gt_masks) == 0:
                        # If there are no ground truths, we can't score anything.
                        # Appending 0 for no predictions is also fine here if needed,
                        # but continuing is often cleaner. Let's keep your logic.
                        if len(pred_masks) == 0:
                            dice_scores.append(torch.tensor(0.0, device=device)) # Or handle as you see fit
                        continue

                    # If there are GTs but no predictions, we score 0 for each GT
                    if len(pred_masks) == 0:
                        for _ in gt_masks:
                            dice_scores.append(torch.tensor(0.0, device=device))
                        continue

                    # Main logic when both GT and predictions exist
                    for gt_mask in gt_masks:
                        # FIX: Initialize best_dice as a tensor on the correct device
                        best_dice = torch.tensor(0.0, device=device)
                        for pred_mask in pred_masks:
                            dice = dice_coefficient(pred_mask, gt_mask)
                            # torch.max is safer for tensors
                            best_dice = torch.max(best_dice, dice)
                        dice_scores.append(best_dice)

        # This line will now work correctly
        # Handle the case where dice_scores might be empty
        if not dice_scores:
            mean_dice = 0.0
        else:
            mean_dice = torch.stack(dice_scores).mean().item()

        print(f"Validation Dice: {mean_dice:.4f}")