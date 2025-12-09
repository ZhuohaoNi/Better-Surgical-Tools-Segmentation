# From U-Net to Transformers: Robustness and Sim-to-Real Transfer in Surgical Vision

This repository contains the official implementation for the project report: **"From U-Net to Transformers: Robustness and Sim-to-Real Transfer in Surgical Vision"**.

The code is based on the [CaRTS repository](https://github.com/hding2455/CaRTS).

## Abstract
While modern surgical tool segmentation models achieve high accuracy on clean data, their performance degrades drastically under real-world artifacts like smoke, bleeding, and low lighting. This study investigates architectural robustness against these corruptions and the transferability of synthetic training to real surgical videos. We evaluated four architectures (U-Net, U-Net++, Mask2Former, and SAM) on the [SegSTRONG-C benchmark](https://github.com/hding2455/CaRTS). Our results demonstrate that transformer-based architectures (Mask2Former) significantly outperform classical CNNs under corruption, achieving the highest Dice scores across most degraded domains.

## Supported Models
This repository supports training and evaluation of the following models on the SegSTRONG-C dataset:
- **U-Net**
- **U-Net++**
- **Mask2Former**

## Installation

### Docker (Recommended)
We provide a Dockerfile for easy environment setup.

1.  **Build the Docker image:**
    ```bash
    cd docker
    docker build ./ -t segstrongc:latest
    ```

2.  **Run the container:**
    ```bash
    ./start_docker.sh
    ```
    *Note: Ensure your dataset is located at `datasets/` or modify `start_docker.sh` to point to your data directory.*

## Usage

### Training
To train a model, use the `train.py` script with the corresponding configuration file.

**Available Configs:**
- `UNet_SegSTRONGC`
- `UNetPlusPlus_SegSTRONGC`
- `Mask2Former_SegSTRONGC`

**Example:**
```bash
python train.py --config Mask2Former_SegSTRONGC
```

### Evaluation
To evaluate a trained model on specific domains (Regular, Smoke, Blood, Background Change, Low Brightness):

**Example:**
```bash
python validate.py \
    --config Mask2Former_SegSTRONGC \
    --model_path checkpoints/mask2former_segstrongc_fulldataset/model_39.pth \
    --test True \
    --domain smoke \
    --save_dir results/smoke
```

### SAM3
To use SAM3 foundation model for inference, run

**Example:**
```bash
python ./sam3_validate.py --config SAM3_SegSTRONGC --test True --domain bg_change
```

### YOLO
To use YOLOv11 model fine-tuning, run

**Example:**
```bash
python yolo/train.py
```

### Visualization
We provide a script to generate side-by-side comparisons of all models across all domains.

**Generate Visualizations:**
```bash
python scripts/save_visualizations.py
```
This will generate images in `visualizations/Combined/` for each domain.

## Authors

- Zhuohao Ni  
- Bruce (Yuqian) Zhang  
