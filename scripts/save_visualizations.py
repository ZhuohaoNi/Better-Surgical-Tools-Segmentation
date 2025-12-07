import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import sys

# Add parent directory to path to allow imports from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs import config_dict
from datasets import dataset_dict
from CaRTS import build_model
import torch.nn.functional as F

def segmentation_overlay(image, gt_mask, pred_mask):
    # image: (H, W, 3) uint8
    # gt_mask: (H, W) 0 or 1
    # pred_mask: (H, W) 0 or 1
    
    overlay = image.copy()
    if overlay.max() <= 1.0:
        overlay = (overlay * 255).astype(np.uint8)
        
    # Create colored masks
    # TP: Green, FP: Red, FN: Blue
    tp = (gt_mask == 1) & (pred_mask == 1)
    fp = (gt_mask == 0) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)
    
    # Apply colors
    overlay[tp] = [0, 255, 0]
    overlay[fp] = [255, 0, 0]
    overlay[fn] = [0, 0, 255]
    
    return overlay

def load_model(config_name, checkpoint_path, device):
    cfg = config_dict[config_name]
    model = build_model(cfg.model, device)
    model.load_parameters(checkpoint_path)
    model.eval()
    return model

def save_combined_visualizations(models_info, output_root="visualizations/Combined", num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    domains = ['regular', 'smoke', 'blood', 'bg_change', 'low_brightness']
    
    # Load all models
    models = {}
    for name, info in models_info.items():
        print(f"Loading {name}...")
        models[name] = load_model(info['config'], info['checkpoint'], device)

    for domain in domains:
        print(f"Processing domain: {domain}")
        domain_output_dir = os.path.join(output_root, domain)
        if not os.path.exists(domain_output_dir):
            os.makedirs(domain_output_dir)

        # Use UNet config for dataset loading
        cfg = config_dict['UNet_SegSTRONGC']
        cfg.test_dataset['args']['domains'] = [domain]
        dataset = dataset_dict[cfg.test_dataset['name']](**(cfg.test_dataset['args']))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        
        for i, (image, gt) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            image = image.to(device)
            gt = gt.to(device)
            
            # Prepare base image for visualization
            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            gt_np = gt[0, 0].cpu().numpy() > 0.5
            
            # Create GT image (White)
            h, w = gt_np.shape
            gt_img = np.zeros((h, w, 3), dtype=np.uint8)
            gt_img[gt_np] = [255, 255, 255]
            
            # Collect overlays
            overlays = []
            
            # 1. Mask2Former
            m2f_model = models['Mask2Former']
            m2f_input = F.interpolate(image, size=(288, 480), mode='bilinear', align_corners=False)
            with torch.no_grad():
                m2f_pred = m2f_model({'image': m2f_input})['pred']
            m2f_pred = F.interpolate(m2f_pred, size=(270, 480), mode='bilinear', align_corners=False)
            m2f_pred_np = m2f_pred[0, 0].cpu().numpy() > 0.5
            overlays.append(segmentation_overlay(img_uint8, gt_np, m2f_pred_np))
            
            # 2. UNet
            unet_model = models['UNet']
            with torch.no_grad():
                unet_pred = unet_model({'image': image})['pred']
            unet_pred_np = unet_pred[0, 0].cpu().numpy() > 0.5
            overlays.append(segmentation_overlay(img_uint8, gt_np, unet_pred_np))
            
            # 3. UNet++
            unetpp_model = models['UNetPlusPlus']
            unetpp_input = F.interpolate(image, size=(288, 480), mode='bilinear', align_corners=False)
            with torch.no_grad():
                unetpp_pred = unetpp_model({'image': unetpp_input})['pred']
            unetpp_pred = F.interpolate(unetpp_pred, size=(270, 480), mode='bilinear', align_corners=False)
            unetpp_pred_np = unetpp_pred[0, 0].cpu().numpy() > 0.5
            overlays.append(segmentation_overlay(img_uint8, gt_np, unetpp_pred_np))
            
            # Combine: Input | GT | Mask2Former | UNet | UNet++
            combined = np.concatenate([img_uint8, gt_img] + overlays, axis=1)
            
            save_path = os.path.join(domain_output_dir, f"combined_{domain}_{i}.png")
            cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"Saved {save_path}")

if __name__ == "__main__":
    models_info = {
        'Mask2Former': {
            'config': 'Mask2Former_SegSTRONGC',
            'checkpoint': 'checkpoints/mask2former_segstrongc_fulldataset/model_39.pth'
        },
        'UNet': {
            'config': 'UNet_SegSTRONGC',
            'checkpoint': 'checkpoints/unet_segstrongc_fulldataset/model_39.pth'
        },
        'UNetPlusPlus': {
            'config': 'UNetPlusPlus_SegSTRONGC',
            'checkpoint': 'checkpoints/unetplusplus_segstrongc_fulldataset/model_39.pth'
        }
    }
    
    save_combined_visualizations(models_info)
