#!/bin/bash
# Stage 1: Font-Augmented Diffusion (FAD) Training Script
# Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval

# Configuration
NUM_GPUS=8
MASTER_PORT=1234
CONFIG="configs/stage1/train.yaml"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "========================================"
echo "Stage 1: FAD Training"
echo "Project Root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "========================================"

# Create necessary directories
mkdir -p weights/stage1
mkdir -p outputs/stage1/logs
mkdir -p outputs/stage1/validation

# Run distributed training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    src/stage1/train.py \
    --config ${CONFIG}

echo "Training completed!"

