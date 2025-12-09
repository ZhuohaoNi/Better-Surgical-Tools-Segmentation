import argparse
from ultralytics import YOLO
from pathlib import Path

def run_training(data_yaml, project_dir, run_name, model_name='yolov11s.pt', epochs=200, batch_size=4, img_size=640):
    """
    Fine-tunes a YOLOv11 model and saves results to a specified directory.
    """
    print(f"Starting fine-tuning with the following parameters:")
    print(f"  - Starting Model: {model_name}")
    print(f"  - Dataset YAML: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {img_size}")
    print(f"  - Results will be saved in: {Path(project_dir) / run_name}")
    print("-" * 30)

    # Load the specified pre-trained YOLO model
    model = YOLO(model_name)

    # Start the fine-tuning process using the custom project and name
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_dir, # This is the main output directory
        name=run_name,        # This is the specific experiment subdirectory
        workers=1
    )
    
    # The full path to the results directory
    save_directory = Path(results.save_dir)
    best_model_path = save_directory / 'weights/best.pt'
    
    print("\n" + "=" * 30)
    print("TRAINING COMPLETE")
    print(f"Results, logs, and weights saved to: {save_directory.resolve()}")
    print(f"The BEST checkpoint is saved as: {best_model_path.resolve()}")
    print(f"The FINAL checkpoint is saved as: {(save_directory / 'weights/last.pt').resolve()}")
    print("=" * 30 + "\n")
    
    return best_model_path

def run_vanilla_training(data_yaml, project_dir, run_name, model_name='yolov11s.pt', epochs=200, batch_size=4, img_size=640):
    """
    Fine-tunes a YOLOv11 model with DATA AUGMENTATION DISABLED.
    """
    print(f"Starting fine-tuning with the following parameters:")
    print(f"  - Starting Model: {model_name}")
    print(f"  - Dataset YAML: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {img_size}")
    print(f"  - Augmentation: DISABLED") 
    print(f"  - Results will be saved in: {Path(project_dir) / run_name}")
    print("-" * 30)

    # Load the specified pre-trained YOLO model
    model = YOLO(model_name)

    # Start the fine-tuning process
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_dir,
        name=run_name,
        workers=1,
        
        # --- DISABLE AUGMENTATION SETTINGS ---
        hsv_h=0.0,      # Hue adjustment
        hsv_s=0.0,      # Saturation adjustment
        hsv_v=0.0,      # Value (brightness) adjustment
        degrees=0.0,    # Rotation
        translate=0.0,  # Translation (shift)
        scale=0.0,      # Scale gain (zoom)
        shear=0.0,      # Shear
        perspective=0.0,# Perspective distortion
        flipud=0.0,     # Flip Up-Down
        fliplr=0.0,     # Flip Left-Right
        mosaic=0.0,     # Mosaic (combining 4 images) - CRITICAL to disable
        mixup=0.0,      # MixUp (blending images)
        copy_paste=0.0, # Copy-Paste segments
        erasing=0.0,    # Random erasing
        crop_fraction=1.0 # Disable random cropping (1.0 means use 100% of image)
    )
    
    # The full path to the results directory
    save_directory = Path(results.save_dir)
    best_model_path = save_directory / 'weights/best.pt'
    
    print("\n" + "=" * 30)
    print("TRAINING COMPLETE")
    print(f"Results, logs, and weights saved to: {save_directory.resolve()}")
    print(f"The BEST checkpoint is saved as: {best_model_path.resolve()}")
    print(f"The FINAL checkpoint is saved as: {(save_directory / 'weights/last.pt').resolve()}")
    print("=" * 30 + "\n")
    
    return best_model_path

def run_evaluation(best_model_path, data_yaml):
    """
    Evaluates the fine-tuned model on the validation set.
    """
    if not best_model_path.exists():
        print(f"Error: Model weights not found at {best_model_path}")
        return

    print(f"Starting evaluation on the validation set using model: {best_model_path}...")
    
    model = YOLO(best_model_path)
    
    metrics = model.val(data=data_yaml, split='val')
    
    print("\n" + "=" * 30)
    print("EVALUATION COMPLETE")
    print("Validation Metrics:")
    print(f"  - mAP50 (mean Average Precision @ IoU=0.50): {metrics.box.map50:.4f}")
    print(f"  - mAP50-95 (mean Average Precision @ IoU=0.50-0.95): {metrics.box.map:.4f}")
    print(f"  - Precision: {metrics.box.p[0]:.4f}")
    print(f"  - Recall: {metrics.box.r[0]:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate a YOLOv11 model.")
    parser.add_argument('--data-yaml', type=str, default='data/yolo_cholecseg20k/dataset.yaml', help="Path to your dataset's .yaml file.")
    parser.add_argument('--model-name', type=str, default='yolo/pre-train/yolo11s.pt', help="Starting pre-trained model (e.g., yolov11n.pt, yolov11s.pt, yolov11m.pt).")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--project-dir', type=str, default='yolo/fine-tune', help="The main directory to save all training runs.")
    parser.add_argument('--run-name', type=str, default='run1', help="The specific name for this training run (will be a subdirectory in project-dir).")
    parser.add_argument('--vanilla', action='store_true', help="If set, disables all data augmentation during training.")

    args = parser.parse_args()
    
    # Create the project directory if it doesn't exist
    Path(args.project_dir).mkdir(parents=True, exist_ok=True)
    
    if args.vanilla:
        # --- Step 1: Train the model with augmentation disabled ---
        best_model_weights_path = run_vanilla_training(
            data_yaml=args.data_yaml,
            project_dir=args.project_dir,
            run_name=args.run_name,
            model_name=args.model_name,
            epochs=args.epochs
        )
    else:
        # --- Step 1: Train the model ---
        best_model_weights_path = run_training(
            data_yaml=args.data_yaml,
            project_dir=args.project_dir,
            run_name=args.run_name,
            model_name=args.model_name,
            epochs=args.epochs
        )
    
    # --- Step 2: Evaluate the best model ---
    if best_model_weights_path:
        run_evaluation(best_model_weights_path, args.data_yaml)