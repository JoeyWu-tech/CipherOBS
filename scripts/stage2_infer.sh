#!/bin/bash
# Stage 2: Stroke Refinement (SR) Inference Script
# Usage: bash scripts/stage2_infer.sh

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Navigate to project root
cd "$(dirname "$0")/.."

# Run inference
python src/stage2/infer.py --config configs/stage2/infer.yaml

