import numpy as np
import cv2
import torch
import time
import logging
import matplotlib.pyplot as plt
import os
from svimg import save_image_unique
from svimg import tensor_to_image
from torch.optim.lr_scheduler import (
    StepLR, 
    ExponentialLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau, 
    CyclicLR, 
    OneCycleLR,
    CosineAnnealingWarmRestarts
)

from pytorchtools import EarlyStopping

class Trainer2:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 mine_epoch: int = 25,       # start mining after this epoch
                 max_mine_train: int = 22,
                 max_mine_val: int = 11,
                 min_error_threshold: int = 150  
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.all_mean_activations = []
        self.all_max_activations = []
        self.lr_scheduler = None

        #for mcunet
        self.lr_scheduler = CyclicLR(
            optimizer,
            
            #1e-4 to 1e-3 (0.0001 to 0.001) - Standard range
            # 1e-4 to 1e-3 (0.0001 to 0.001) - Standard range
            #3e-4 (0.0003) - Most common default
            #1e-3 (0.001) - Default in many frameworks but often too high
            #1e-5 to 1e-4 - For fine-tuning pre-trained models
            
            
            base_lr=1e-4,      # Minimum LR
            max_lr=1e-3,       # Maximum LR
            step_size_up=500, # Gradual increase for 2000 iterations
            step_size_down=500, # Gradual decrease for 2000 iterations
            mode='triangular', # Linear up and down
            cycle_momentum=False
        ) 
        
        #total_steps = num_epochs * (train_dataset_size // batch_size)
        #total_steps = 222 * (660 // 2)
        # #for mctransunet
        # self.lr_scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr= (1e-3) * 3,      # Peak LR 0.003
        #     total_steps=total_steps,  # Total training steps
        #     pct_start=0.3,    # 30% high LR, then decay
        #     anneal_strategy='cos',  # Cosine decay
        #     cycle_momentum=False  # Keep False for AdamW
        # )
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook

        self.training_loss = []
        self.validation_loss = []
        self.learning_rate = []
        
        # for analyzing weight per independent loss
        self.weights_history = []

        
        #for cascadeunet
        self.epoch_train_preds = []
        self.epoch_train_targets = []
        self.epoch_val_preds = []
        self.epoch_val_targets = []
        self.best_pred_epoch = 50
        
        # Stage-2 sample collection params
        # self.mine_epoch = 25       # start mining after this epoch
        # self.max_mine_train = 660
        # self.max_mine_val = 330
        # self.min_error_threshold = 150  # tune later

        self.mine_epoch = mine_epoch       # start mining after this epoch
        self.max_mine_train = max_mine_train
        self.max_mine_val = max_mine_val
        self.min_error_threshold = min_error_threshold  # tune later
        

        
        # Set up a logger
        self.logger = logging.getLogger('Trainer2')
        self.logger.setLevel(logging.DEBUG)
        logging.basicConfig(filename='./log/training.log', level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

    
    
    def run_trainer(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        
        
        #ここでearlystoppingの打ち切り回数設定
        early_stopping = EarlyStopping(patience = 50,verbose = True)
        
        
        
        try:
            for i in progressbar:
                """Epoch counter"""
                self.epoch += 1  # epoch counter

                """Training block"""
                self._train()

                """Validation block"""
                if self.validation_DataLoader is not None:
                    self._validate()
                
                if isinstance(self.lr_scheduler, StepLR):
                    #print(f"Using StepLR. Epoch: {self.epoch + 1}")
                    self.lr_scheduler.step()

                elif isinstance(self.lr_scheduler, ExponentialLR):
                    #print(f"Using ExponentialLR. Epoch: {self.epoch + 1}")
                    self.lr_scheduler.step()

                elif isinstance(self.lr_scheduler, CosineAnnealingLR):
                    #print(f"Using CosineAnnealingLR. Epoch: {self.epoch + 1}")
                    self.lr_scheduler.step()

                elif isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    #print(f"Using ReduceLROnPlateau. Epoch: {self.epoch + 1}")
                    self.lr_scheduler.step(self.validation_loss[i])  # Pass the loss for ReduceLROnPlateau
                elif isinstance(self.lr_scheduler, CosineAnnealingWarmRestarts):
                    #self.lr_scheduler.step(self.epoch + 1)
                    self.lr_scheduler.step(self.epoch + 1)
                    
                elif isinstance(self.lr_scheduler, CyclicLR):
                    self.lr_scheduler.step()
                    #print(f"Using CyclicLR (updated per batch). Epoch: {self.epoch + 1}")

                elif isinstance(self.lr_scheduler, OneCycleLR):
                    self.lr_scheduler.step()

                else:
                    print("Unknown scheduler. No specific action taken.")
                    
                    
                # """Learning rate scheduler block"""
                # if self.lr_scheduler is not None:
                #     if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                #         self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                #     else:
                #         self.lr_scheduler.step(self.epoch+1)  # StepLR/Cosine
                        
                print('train_losses',self.training_loss[i])        
                print('val_losses',self.validation_loss[i])
                print('lr', self.learning_rate[i])
                
                ## for ensembleinspiredloss
                #print(self.criterion.get_loss_breakdown())
                

                # Logging
                self.logger.info(f'Epoch: {i + 1}')
                self.logger.info(f'Training Loss: {self.training_loss[-1]}')
                self.logger.info(f'Validation Loss: {self.validation_loss[-1]}')
                self.logger.info(f'Learning Rate: {self.learning_rate[-1]}')

                early_stopping(self.validation_loss[i],self.model)
        
                if early_stopping.early_stop:
                    print(f"early stopping epoch:",i)
                    self.logger.info(f"Early stopping epoch: {i}")
                    
                    # self.model.load_state_dict(torch.load('checkpoint.pt'))
                    # print("Loaded best model weights from early stopping checkpoint")   
                    
                    break
            
            # training and validation loss plot
            # analysis on the training is done here            
            fig, ax = plt.subplots(figsize=(7, 5))  # Create figure and axes
            ax.plot(self.training_loss, label='Training Loss', color='blue')
            ax.plot(self.validation_loss, label='Validation Loss', color='orange')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training vs Validation Loss')
            ax.legend()
            ax.grid(True)
            
            

        except Exception as e:  
            print("An error occurred during training/validation:")
            print(e)
            
            import traceback
            traceback.print_exc()
            self.logger.exception("An error occurred during training/validation:")
            
            raise  # Re-raise the exception to see the full traceback
                    
             
        return self.training_loss, self.validation_loss, self.learning_rate, fig, self.epoch_train_preds, self.epoch_train_targets, self.epoch_val_preds, self.epoch_val_targets

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        #print("start train model!")
        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        #for cascade
        epoch_preds = []
        epoch_targets = []
        
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
        
        
        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            
            # print(f'Train Input size: {input.size()}')
            # print(f'Train Target size: {target.size()}')
            
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            
            
            if self.epoch >= self.mine_epoch:
                probs = torch.softmax(out, dim=1).detach().cpu()  # [B, C, H, W]
                preds = torch.argmax(probs, dim=1)                # [B, H, W]
                targets_cpu = target.detach().cpu()

                for b in range(preds.shape[0]):
                    pred_b = preds[b]
                    target_b = targets_cpu[b]
                    prob_b = probs[b]    # use probabilities for stage-2 input

                    wrong = (pred_b != target_b)                  # pixel-wise
                    fg    = (target_b != 0)                       # foreground mask
                    hard_errors = (wrong & fg)

                    if hard_errors.sum() >= self.min_error_threshold:
                        if len(self.epoch_train_preds) < self.max_mine_train:
                            # Keep a batch dimension so later concatenation yields [N, C, H, W] and [N, H, W]
                            self.epoch_train_preds.append(prob_b.unsqueeze(0))    # [1,C,H,W]
                            self.epoch_train_targets.append(target_b.unsqueeze(0)) # [1,H,W]
                            
                            print("Train preds for stage 2: " + str(len(self.epoch_train_preds)))

                            if len(self.epoch_train_preds) >= self.max_mine_train:
                                break

            
            # for combined loss with learnable weight
            loss = self.criterion(out, target)  # calculate loss
            #print("Learned loss weights")
            #print(weights)
            #for other losses
            #loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
        
        # #cascade
        # if self.epoch == self.best_pred_epoch:
        #     self.epoch_train_preds = torch.cat(epoch_preds)
        #     self.epoch_train_targets = torch.cat(epoch_targets)

        
        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close() 
        


    def _validate(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        
        #for cascade
        epoch_val_preds = []
        epoch_val_targets = []
        
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            # print(f'Valid Input size: {input.size()}')
            # print(f'Valid Target size: {target.size()}')
            
            with torch.no_grad():
                out = self.model(input)
                
                #cascade
                if self.epoch >= self.mine_epoch:
                    probs = torch.softmax(out, dim=1).detach().cpu()  # [B,C,H,W]
                    preds = torch.argmax(probs, dim=1)                # [B,H,W]
                    targets_cpu = target.detach().cpu()

                    for b in range(preds.shape[0]):
                        pred_b = preds[b]
                        target_b = targets_cpu[b]
                        prob_b = probs[b]  # softmax output for stage-2

                        wrong = (pred_b != target_b)
                        fg    = (target_b != 0)
                        hard_errors = (wrong & fg)

                        if hard_errors.sum() >= self.min_error_threshold:
                            if len(self.epoch_val_preds) < self.max_mine_val:
                                # Keep a batch dimension to match training preds format
                                self.epoch_val_preds.append(prob_b.unsqueeze(0))
                                self.epoch_val_targets.append(target_b.unsqueeze(0))
                                print("Val preds for stage 2: " + str(len(self.epoch_train_preds)))

                            # stop mining but keep validating
                            if len(self.epoch_train_preds) >= self.max_mine_val:
                                break


                
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
                
                # Optional visualization for first batch/image
                ###
                # if i == 1:
                #     # limited by number of images in batch
                #     # in my case batch size is 2 so only 0 and 1
                #     image_idx = 0
                #     logits = out[image_idx]  # [C, H, W] - logits for first image
                #     probs = torch.softmax(logits, dim=0)  # normalize across classes (C)

                #     # print("Probabilities shape")
                #     # print(probs.shape)       # [C, H, W]
                #     # print("Sum across classes around ~1 per pixel")
                #     # print(probs.sum(dim=0))  # each pixel's class probs ≈ 1

                #     num_classes = probs.shape[0]

                #     plt.figure(figsize=(24, 4))
                #     for c in range(num_classes):
                #         plt.subplot(1, num_classes, c + 1)
                #         plt.imshow(probs[c].cpu(), cmap='viridis')
                #         plt.title(f'Prob - Class {c}')
                #         plt.axis('off')
                #     plt.show()

                #     # Get predicted class per pixel
                #     pred_mask = probs.argmax(dim=0)  # [H, W]

                #     # Ground truth comparison (assuming target shape [B, H, W])
                #     error_map = (pred_mask.cpu() != target[image_idx].cpu()).float()

                #     plt.imshow(error_map, cmap='Reds')
                #     plt.title('Misclassified pixels')
                #     plt.show()

                ####
        # if self.epoch == self.best_pred_epoch:
        #     self.epoch_val_preds = torch.cat(epoch_val_preds)
        #     self.epoch_val_targets = torch.cat(epoch_val_targets)

        self.validation_loss.append(np.mean(valid_losses))
        batch_iter.close()

    def plot_mined_samples(self, n=10, split='train', save_path: str = None):
        """Plot up to `n` mined samples for `split` in {'train','val'}.

        Each row shows: [predicted mask, ground truth]. Predicted mask is
        obtained by argmax over probability channels.
        Returns the path to the saved figure or None if nothing to plot.
        """
        import torch

        if split == 'train':
            preds_list = self.epoch_train_preds
            targets_list = self.epoch_train_targets
        else:
            preds_list = self.epoch_val_preds
            targets_list = self.epoch_val_targets

        if len(preds_list) == 0 or len(targets_list) == 0:
            print(f"No mined {split} samples to plot.")
            return None

        # Concatenate collected tensors (they were stored with a leading batch dim)
        try:
            preds = torch.cat(preds_list, dim=0)  # [N, C, H, W]
            targets = torch.cat(targets_list, dim=0)  # [N, H, W]
        except Exception as e:
            print("Error concatenating mined samples:", e)
            return None

        N = preds.shape[0]
        k = min(n, N)

        pred_masks = torch.argmax(preds, dim=1).cpu().numpy()  # [N, H, W]
        targets_np = targets.cpu().numpy()  # [N, H, W]

        # Prepare figure with k rows and 2 columns
        fig, axes = plt.subplots(k, 2, figsize=(6, 3 * k))

        if k == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(k):
            ax_pred = axes[i, 0]
            ax_gt = axes[i, 1]

            ax_pred.imshow(pred_masks[i], cmap='tab20')
            ax_pred.set_title(f'{split} pred #{i}')
            ax_pred.axis('off')

            ax_gt.imshow(targets_np[i], cmap='tab20')
            ax_gt.set_title('ground truth')
            ax_gt.axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join('./log', f'mined_{split}_samples_epoch{self.epoch}.png')

        try:
            fig.savefig(save_path)
            plt.close(fig)
            print(f"Saved mined {split} samples figure to: {save_path}")
            return save_path
        except Exception as e:
            print("Failed to save mined samples figure:", e)
            plt.close(fig)
            return None
