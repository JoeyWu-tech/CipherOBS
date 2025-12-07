#!/bin/bash
# Stage 1: Font-Augmented Diffusion (FAD) Inference Script
# Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval

# Configuration
NUM_GPUS=8
MASTER_PORT=1235
CONFIG="configs/stage1/infer.yaml"

# Optional: Override seed from command line
SEED=${1:-61}

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "========================================"
echo "Stage 1: FAD Inference"
echo "Project Root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "Seed: ${SEED}"
echo "========================================"

# Create output directory
mkdir -p outputs/stage1/results

# Run distributed inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=${MASTER_PORT} \
    src/stage1/infer.py \
    --config ${CONFIG} \
    --seed ${SEED}

echo "Inference completed!"

