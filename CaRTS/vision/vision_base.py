import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class VisionBase(nn.Module):
    def __init__(self, params, device):
        super(VisionBase, self).__init__()
        self.train_params = params['train_params']
        self.device = device

    def get_feature_map(self, x):
        raise NotImplementedError()

    def forward(self, x, return_loss=False):
        raise NotImplementedError()

    def load_parameters(self, load_path):
        self.load_state_dict(torch.load(load_path, map_location=self.device)['state_dict'])

    def train_epochs(self, train_dataloader, validation_dataloader, load_path=None):
        train_params = self.train_params
        optimizer = train_params['optimizer']
        lr_scheduler = train_params['lr_scheduler']
        max_epoch_number = train_params['max_epoch_number']
        save_interval = train_params['save_interval']
        save_path = train_params['save_path'] 
        log_interval = train_params['log_interval']
        perturbation = train_params['perturbation']
        device = self.device
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if load_path is not None:
            checkpoint = torch.load(load_path, map_location=device)
            state_dict = checkpoint['state_dict']
            self.load_state_dict(state_dict)
            current_epoch_numbers = checkpoint['current_epoch_numbers']
            loss_plot = checkpoint['loss_plot']
            optimizer = optimizer["optim_class"](self.parameters(), **(optimizer["args"]))
            lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, last_epoch=current_epoch_numbers, **(lr_scheduler["args"]))
        else:
            optimizer = optimizer["optim_class"](self.parameters(), **(optimizer["args"]))
            lr_scheduler = lr_scheduler["lr_scheduler_class"](optimizer, **(lr_scheduler["args"]))
            current_epoch_numbers = 0
            loss_plot = []
        
        # Initialize GradScaler for mixed precision training
        scaler = GradScaler()

        for e in range(current_epoch_numbers, max_epoch_number):
            self.train()
            running_loss = 0
            start = time.time()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                       desc=f'Epoch {e+1}/{max_epoch_number}')
            for i, (image, gt) in pbar:
                self.zero_grad()
                data = {}
                if perturbation is not None:
                    image = perturbation(image/255) * 255
                data['image'] = image.to(device=device)
                data['gt'] = gt.to(device=device)
                
                # Mixed precision forward pass
                with autocast():
                    pred, loss = self.forward(data, return_loss=True)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                elapsed = time.time() - start
                if (i+1) % log_interval == 0:
                    loss_plot.append(running_loss / (i+1))
                
                # Update progress bar with loss and speed
                pbar.set_postfix({
                    'Loss': f'{running_loss / (i+1):.6f}',
                    'Speed': f'{(i+1)*pred.size(0) / elapsed:.1f} it/s'
                })
            
            print(f"Epoch {e+1}/{max_epoch_number} - Loss: {running_loss / (i+1):.6f} - "
                  f"Speed: {(i+1)*pred.size(0) / elapsed:.1f} it/s")
            lr_scheduler.step()
            if (e+1) % save_interval == 0:
                save_dict = {}
                save_dict['state_dict'] = self.state_dict()
                save_dict['current_epoch_numbers'] = e
                save_dict['loss_plot'] = loss_plot
                torch.save(save_dict, os.path.join(save_path,"model_"+str(e)+".pth"))
                self.eval()
                validation_loss = 0
                start = time.time()
                for i, (image, gt) in enumerate(validation_dataloader):
                   data['image'] = image.to(device=device)
                   data['gt'] = gt.to(device=device)
                   pred, loss = self.forward(data, return_loss=True)
                   validation_loss += loss.item()
                elapsed = time.time() - start
                print("Validation at epch : %d Validation Loss: %f iteration per Sec: %f" %
                            (e, validation_loss / (i+1), (i+1) / elapsed))
        return loss_plot