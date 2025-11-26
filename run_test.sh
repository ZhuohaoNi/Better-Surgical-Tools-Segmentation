#!/bin/bash

# Configuration
CONFIG="Mask2Former_SegSTRONGC"
MODEL_PATH="/workspace/code/checkpoints/mask2former_segstrongc_fulldataset/model_39.pth"
SAVE_BASE_DIR="/workspace/code/results"

# List of domains to test
DOMAINS=("regular" "smoke" "blood" "bg_change" "low_brightness")

echo "Starting testing on all domains..."
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG"
echo "=================================="

for domain in "${DOMAINS[@]}"
do
    echo ""
    echo "Testing domain: $domain"
    echo "--------------------------------"
    
    SAVE_DIR="$SAVE_BASE_DIR/$domain"
    
    python validate.py \
        --config "$CONFIG" \
        --model_path "$MODEL_PATH" \
        --test True \
        --domain "$domain" \
        --save_dir "$SAVE_DIR" \
        --batch_size 4 \
        --num_workers 2
    
    if [ $? -eq 0 ]; then
        echo "✓ $domain completed successfully"
    else
        echo "✗ $domain failed"
    fi
done

echo ""
echo "=================================="
echo "All tests completed!"
echo "Results saved in: $SAVE_BASE_DIR"


# CONFIG="UNet_SegSTRONGC"
# MODEL_PATH="/workspace/code/checkpoints/unet_segstrongc_fulldataset/model_39.pth"
# CONFIG="UNetPlusPlus_SegSTRONGC"
# MODEL_PATH="/workspace/code/checkpoints/unetplusplus_segstrongc_fulldataset/model_39.pth"

# # List of domains to test
# DOMAINS=("regular" "smoke" "blood" "bg_change" "low_brightness")

# echo "Starting testing on all domains..."
# echo "Model: $MODEL_PATH"
# echo "Config: $CONFIG"
# echo "=================================="

# for domain in "${DOMAINS[@]}"
# do
#     echo ""
#     echo "Testing domain: $domain"
#     echo "--------------------------------"
    
#     SAVE_DIR="$SAVE_BASE_DIR/$domain"
    
#     python validate.py \
#         --config "$CONFIG" \
#         --model_path "$MODEL_PATH" \
#         --test True \
#         --domain "$domain" \
#         --save_dir "$SAVE_DIR" \
#         --batch_size 8 \
#         --num_workers 4
    
#     if [ $? -eq 0 ]; then
#         echo "✓ $domain completed successfully"
#     else
#         echo "✗ $domain failed"
#     fi
# done

# echo ""
# echo "=================================="
# echo "All tests completed!"
# echo "Results saved in: $SAVE_BASE_DIR"