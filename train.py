from torch.utils.data import DataLoader
import numpy as np
from configs import config_dict
from datasets import dataset_dict
import torch
import time
import argparse
import os
from CaRTS import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Name of the configuration file.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model checkpoint file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = config_dict[args.config]
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
        # Enable TF32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True # Optimize for fixed input size
    else:
        print("WARNING: Using CPU! CUDA is not available.")
        device = torch.device("cpu")

    train_dataset = dataset_dict[cfg.train_dataset['name']](**(cfg.train_dataset['args']))
    batch_size = cfg.train_dataset.get('batch_size', 64)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                   num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    validation_dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False,
                                       num_workers=4, pin_memory=True)
    model = build_model(cfg.model, device)
    
    # Compile model for speedup (PyTorch 2.0+)
    if use_gpu:
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
    
    if args.model_path is None:
        loss_plot = model.train_epochs(train_dataloader, validation_dataloader) 
    else:
        model.load_parameters(args.model_path)
        loss_plot = model.train_epochs(train_dataloader, validation_dataloader) 
