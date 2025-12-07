#!/bin/bash
# Stage 2: Stroke Refinement (SR) Training Script
# Usage: bash scripts/stage2_train.sh

# Set CUDA devices (modify as needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Navigate to project root
cd "$(dirname "$0")/.."

# Run training with accelerate
accelerate launch \
    --config_file configs/acc_configs/gpu4.yaml \
    src/stage2/train.py \
    --config configs/stage2/train.yaml

