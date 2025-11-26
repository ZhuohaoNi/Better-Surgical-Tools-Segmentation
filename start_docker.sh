#!/bin/bash

# Set the path to your dataset here
DATA_DIR="/home/peterni/Documents/CaRTS/datasets"


# Run the docker container
# -v $PWD:/workspace/code : Mounts the current directory (your code) into the container, allowing you to edit files locally.
# -v $DATA_DIR:/workspace/data : Mounts your dataset.
# --gpus all : Enables GPU access.
# --shm-size=8g : Increases shared memory size, often needed for PyTorch data loaders.

docker run --rm \
    --gpus all \
    --shm-size=8g \
    -v "$PWD":/workspace/code \
    -v "$DATA_DIR":/workspace/data \
    -it segstrongc:latest \
    /bin/bash
