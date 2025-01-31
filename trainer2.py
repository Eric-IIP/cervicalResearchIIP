import numpy as np
import torch
import logging
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
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        #self.lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        #Experiment No.0
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        #Experiment No.1
        #self.lr_scheduler = scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        #self.lr_scheduler = CosineAnnealingLR(optimizer, T_max=146)
        #Experiment No.2
        # self.lr_scheduler = ReduceLROnPlateau(
        # optimizer,
        # mode='min',          # Monitor validation loss (minimizing it)
        # factor=0.5,          # Reduce LR by half
        # patience=5,          # Wait 5 epochs for improvement
        # threshold=1e-4,      # Improvement threshold
        # cooldown=2,          # Wait 2 epochs after reducing LR
        # min_lr=1e-6,         # Prevent LR from going too low
        # verbose=True         # Print LR updates
        # )
        
        self.lr_scheduler = CyclicLR(
            optimizer,
            base_lr=1e-4,      # Minimum LR
            max_lr=1e-3,       # Maximum LR
            step_size_up=660, # Gradual increase 660 iterations steps per epoch = trainingset length / batch size; 660/2=330 and then a few epoch multiplied (2-4) in this case: 2
            step_size_down=660, # Gradual decrease for 2000 iterations
            mode='triangular', # Linear up and down
            cycle_momentum=False # Set to True for optimizers like SGD with momentum
        )
        
        #total_steps = num_epochs * (train_dataset_size // batch_size)
        # total_steps = 146 * (660 // 2)
        
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
                        
                        
                print('val_losses',self.validation_loss[i])

                # Logging
                self.logger.info(f'Epoch: {i + 1}')
                self.logger.info(f'Training Loss: {self.training_loss[-1]}')
                self.logger.info(f'Validation Loss: {self.validation_loss[-1]}')
                self.logger.info(f'Learning Rate: {self.learning_rate[-1]}')

                early_stopping(self.validation_loss[i],self.model)
        
                if early_stopping.early_stop:
                    print(f"early stopping epoch:",i)
                    self.logger.info(f"Early stopping epoch: {i}")
                    break

        except Exception as e:
            print("An error occurred during training/validation:")
            print(e)
            
            import traceback
            traceback.print_exc()
            self.logger.exception("An error occurred during training/validation:")
            
            raise  # Re-raise the exception to see the full traceback
                    
             
        return self.training_loss, self.validation_loss, self.learning_rate

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
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar

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

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()
