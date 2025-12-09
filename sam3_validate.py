import torch
import numpy as np
import argparse
import os
import time
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image

import sys
import os

# Add the directory containing 'sam3' to the Python path
# This handles the case where the script is in a sibling directory (e.g., SegSTRONGC/CaRTS)
# and sam3 is in another (e.g., SegSTRONGC/sam3).
current_dir = os.path.dirname(os.path.abspath(__file__))
# sam3_root = os.path.abspath(os.path.join(current_dir, "../sam3"))
# if sam3_root not in sys.path:
#     sys.path.append(sam3_root)

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

# Evaluation Imports (Assumed to be available in the environment)
try:
    from configs import config_dict
    from datasets import dataset_dict
    from CaRTS import build_model  # Kept for reference, but we use SAM3
    from CaRTS.evaluation.metrics import dice_scores, normalized_surface_distances
except ImportError as e:
    print(f"Warning: Could not import some evaluation dependencies: {e}")
    print("Ensure 'configs', 'datasets', and 'CaRTS' are in your python path.")

class SAM3Wrapper(torch.nn.Module):
    def __init__(self, model, processor, text_prompt="surgical tool, usually silver grasper, hook, or scissors", device="cuda"):
        super().__init__()
        self.model = model
        self.processor = processor
        self.text_prompt = text_prompt
        self.device = device

    def forward(self, data):
        images = data['image'] # Expecting (B, C, H, W)
        batch_size = images.shape[0]
        preds = []

        for i in range(batch_size):
            # SAM3 processor set_image typically expects a PIL image or numpy array (H, W, C)
            # We need to convert the tensor to a format SAM3 accepts.
            # Assuming images are normalized tensors, we might need to denormalize or just convert.
            # For now, assuming standard (0-1) or (0-255) tensors.
            
            img_tensor = images[i]
            
            # Convert tensor (C, H, W) to numpy (H, W, C)
            # Detach and move to cpu
            img_np = img_tensor.cpu().detach().permute(1, 2, 0).numpy()
            
            # If float 0-1, convert to uint8 0-255
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
                
            # Create PIL Image
            pil_image = Image.fromarray(img_np)

            # Set image
            inference_state = self.processor.set_image(pil_image)
            
            # Set text prompt
            self.processor.set_text_prompt(self.text_prompt, inference_state)
            
            # Extract masks from inference_state
            # The user example suggests: inference_state["masks"] is a list or tensor indexable by object index
            # and inference_state["scores"] determines the number of objects.
            
            masks = inference_state["masks"]
            scores = inference_state["scores"]
            nb_objects = len(scores)
            
            print(f"[DEBUG] Batch {i}: Found {nb_objects} objects.")
            
            final_mask = torch.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32).to(self.device)
            
            if nb_objects > 0:
                for idx in range(nb_objects):
                    obj_mask = masks[idx]
                    
                    # Ensure obj_mask is on correct device
                    if isinstance(obj_mask, torch.Tensor):
                        obj_mask = obj_mask.to(self.device).float()
                    
                    # Squeeze if necessary (e.g. if (1, H, W))
                    if obj_mask.ndim == 3 and obj_mask.shape[0] == 1:
                        obj_mask = obj_mask.squeeze(0)
                        
                    print(f"  [DEBUG] Obj {idx} Mask Shape: {obj_mask.shape}, Range: [{obj_mask.min():.4f}, {obj_mask.max():.4f}], Mean: {obj_mask.mean():.4f}")
                        
                    # Combine (Logical OR / Max)
                    final_mask = torch.max(final_mask, obj_mask)
            
            print(f"[DEBUG] Final Mask Shape: {final_mask.shape}, Range: [{final_mask.min():.4f}, {final_mask.max():.4f}], Unique truncated: {torch.unique(final_mask)[:10]}")

            # Ensure shape is (1, H, W) for output
            final_mask = final_mask.unsqueeze(0)
            
            preds.append(final_mask)

        # Stack predictions: (B, 1, H, W)
        return {'pred': torch.stack(preds).to(self.device), "nb_objects": nb_objects}


def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 Segmentation Validation")
    parser.add_argument("--config", type=str, required=True, help="Name of the config file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to SAM3 checkpoint (optional, defaults to HF)")
    parser.add_argument("--test", type=bool, default=False, help="True for testing, False for validation")
    parser.add_argument("--domain", type=str, default=None, help="Test/Validate domain")
    parser.add_argument("--save_dir", type=str, default=None, help="Path to save model output")
    parser.add_argument("--tau", type=int, default=5, help="Tolerance in normalized surface distance calculation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--text_prompt", type=str, default="surgical tool, usually silver grasper, hook, or scissors", help="Text prompt for SAM3")
    return parser.parse_args()

def evaluate(model, dataloader, device, tau, save_dir=None):
    start = time.time()
    results = []
    dice_tools = []
    nsds = []
    
    # SAM3 model is already in eval mode from build_sam3_image_model usually, but good to ensure.
    # Wrapper is nn.Module, so we can call eval()
    model.eval()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    total_samples = 0
    print(f"Starting evaluation on {len(dataloader)} batches...")

    zero_object_count = 0
    one_object_count = 0
    tw0_object_count = 0
    
    with torch.no_grad():
        for i, (image, gt) in enumerate(dataloader):
            print(f"Iteration: {i}/{len(dataloader)}", end="\r")

            # Verification: Print input image shape
            print(f"Input image shape: {image.shape}")

            data = dict()
            data['image'] = image.to(device=device)
            data['gt'] = gt.to(device=device)
            data['iteration'] = i
            
            # Inference
            result = model(data)
            pred = result['pred']
            nb_obejcts = result['nb_objects']

            if nb_obejcts == 0:
                zero_object_count += 1
            elif nb_obejcts == 1:
                one_object_count += 1
            elif nb_obejcts == 2:
                tw0_object_count += 1
            
            # Verification: Visualize mask every 50 iterations
            if i % 50 == 0 and save_dir is not None:
                vis_path = os.path.join(save_dir, f"mask_vis_{i}.png")
                # pred is (B, 1, H, W), save_image expects float 0-1
                save_image(pred.float(), vis_path)
                print(f"Saved visualization to {vis_path}")
            
            # Process batch
            preds_np = pred.cpu().detach().numpy()
            gts_np = gt.cpu().numpy()
            
            batch_size = preds_np.shape[0]
            total_samples += batch_size

            for b in range(batch_size):
                # Threshold at 0.5
                result = (preds_np[b] > 0.5).squeeze()
                mask = (gts_np[b] > 0.5).squeeze()
                
                # Handle cases where squeeze results in scalar (0-d array) if H,W are 1 (unlikely)
                # or if batch size is 1 and channel is 1.
                
                dice_tool = dice_scores(result, mask)
                nsd = normalized_surface_distances(result, mask, tau)
                dice_tools.append(dice_tool)
                nsds.append(nsd)
                
                if save_dir is not None:
                    results.append(result)
        
    elapsed = time.time() - start
    print("\nEvaluation Complete.")
    print("Samples per Sec: %f" % (total_samples / elapsed))
    print("mean: dice_tool: %f " % (np.mean(dice_tools)))
    print("std: dice_tool: %f " % (np.std(dice_tools)))
    print("mean: nsd: %f" % (np.mean(nsds)))
    print("std: nsd: %f" % (np.std(nsds)))
    print(f"Zero object count: {zero_object_count}, One object count: {one_object_count}, Two object count: {tw0_object_count}")
    
    if save_dir is not None:
        np.save(os.path.join(save_dir, "pred.npy"), results)

if __name__ == "__main__":
    args = parse_args()
    
    # Load Config
    if args.config not in config_dict:
        print(f"Error: Config '{args.config}' not found in config_dict.")
        exit(1)
        
    cfg = config_dict[args.config]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Update domains in config
    if args.domain is not None:
        domain = args.domain
        if args.test:
            if hasattr(cfg, 'test_dataset') and 'args' in cfg.test_dataset:
                cfg.test_dataset['args']['domains'] = [domain]
        else:
             if 'validation_dataset' in cfg and 'args' in cfg.validation_dataset:
                cfg.validation_dataset['args']['domains'] = [domain]
            
    # Load Dataset
    dataset = None
    dataloader = None
    
    print("Loading dataset...")
    if args.test:
        print("Loading test dataset...")
        dataset = dataset_dict[cfg.test_dataset['name']](**(cfg.test_dataset['args']))
    else:
        dataset = dataset_dict[cfg.validation_dataset['name']](**(cfg.validation_dataset['args']))
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"Dataset loaded. Size: {len(dataset)}")

    # Build SAM3 Model
    print("Building SAM3 model...")
    # Determine checkpoint path
    checkpoint_path = args.model_path
    load_from_hf = True
    if checkpoint_path is not None:
        load_from_hf = False
    
    bpe_path = os.path.join("sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    print(f"Using BPE path: {bpe_path}")
    if not os.path.exists(bpe_path):
        if not os.path.exists(bpe_path):
             print(f"Warning: BPE path {bpe_path} does not exist. Model build might fail.")

    sam3_model = build_sam3_image_model(
        bpe_path=bpe_path,
        device=str(device),
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
        enable_segmentation=True
    )
    
    # Build Processor
    processor = Sam3Processor(sam3_model, confidence_threshold=0.5) # Threshold can be tuned
    
    # Wrap Model
    model = SAM3Wrapper(sam3_model, processor, text_prompt=args.text_prompt, device=device)
    
    # Run Evaluation
    evaluate(model, dataloader, device, args.tau, args.save_dir)