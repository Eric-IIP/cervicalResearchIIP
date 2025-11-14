import numpy as np
import cv2
import torch
import time
import logging
import matplotlib.pyplot as plt
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
                 notebook: bool = False
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
        early_stopping = EarlyStopping(patience = 0,verbose = True)
        
        
        
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
            
            loss_names = [
                "Boundary IOU Loss",
                "Boundary Dice Loss",
                "Focal Loss",
                #"Confusion Penalty Loss",
                "Ensemble Inspired Loss",
                "Dice Loss",
                "Cross Entropy Loss"
            ]
            
            weights_arr = np.stack(self.weights_history)
            
            
            fig2, ax2 = plt.subplots(figsize=(10,6))

            # Plot each loss weight with its actual name
            for i in range(weights_arr.shape[1]):
                ax2.plot(weights_arr[:, i], label=loss_names[i])

            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Learned Weight (softmax)')
            ax2.set_title('Learned Loss Weights Over Training')
            ax2.legend()
            ax2.grid(True)

        except Exception as e:  
            print("An error occurred during training/validation:")
            print(e)
            
            import traceback
            traceback.print_exc()
            self.logger.exception("An error occurred during training/validation:")
            
            raise  # Re-raise the exception to see the full traceback
                    
             
        return self.training_loss, self.validation_loss, self.learning_rate, fig, fig2

    def _train(self):

        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        #print("start train model!")
        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                          leave=False)
        
        
        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            
            # print(f'Train Input size: {input.size()}')
            # print(f'Train Target size: {target.size()}')
            
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            # for commbined loss with learnable weight
            loss, weights = self.criterion(out, target, return_weights = True)  # calculate loss
            #print("Learned loss weights")
            #print(weights)
            #for other losses
            #loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
        
        # for combined loss with learnable weight
        self.weights_history.append(weights.cpu().detach().numpy())
        
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
        batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                          leave=False)

        for i, (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            # print(f'Valid Input size: {input.size()}')
            # print(f'Valid Target size: {target.size()}')
            
            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')
                
                # Optional visualization for first batch/image
                ###
                if i == 1:
                    # limited by number of images in batch
                    # in my case batch size is 2 so only 0 and 1
                    image_idx = 0
                    logits = out[image_idx]  # [C, H, W] - logits for first image
                    probs = torch.softmax(logits, dim=0)  # normalize across classes (C)

                    # print("Probabilities shape")
                    # print(probs.shape)       # [C, H, W]
                    # print("Sum across classes around ~1 per pixel")
                    # print(probs.sum(dim=0))  # each pixel's class probs ≈ 1

                    num_classes = probs.shape[0]

                    plt.figure(figsize=(24, 4))
                    for c in range(num_classes):
                        plt.subplot(1, num_classes, c + 1)
                        plt.imshow(probs[c].cpu(), cmap='viridis')
                        plt.title(f'Prob - Class {c}')
                        plt.axis('off')
                    plt.show()

                    # Get predicted class per pixel
                    pred_mask = probs.argmax(dim=0)  # [H, W]

                    # Ground truth comparison (assuming target shape [B, H, W])
                    error_map = (pred_mask.cpu() != target[image_idx].cpu()).float()

                    plt.imshow(error_map, cmap='Reds')
                    plt.title('Misclassified pixels')
                    plt.show()

                ####

        self.validation_loss.append(np.mean(valid_losses))
        batch_iter.close()
