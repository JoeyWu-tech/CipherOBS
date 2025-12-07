#!/bin/bash
# ============================================================================
# OBS Dictionary Retrieval Inference Script
# ============================================================================
# 
# This script performs Oracle Bone Script decipherment via dictionary retrieval.
# 
# Usage:
#   bash scripts/retrieval_infer.sh [OPTIONS]
#
# Options:
#   --config    Path to config file (default: configs/retrieval/infer.yaml)
#   --query     Override query directory
#   --dict      Override dictionary directory  
#   --gpu       GPU ID to use (default: 0)
#
# Example:
#   bash scripts/retrieval_infer.sh --gpu 0
#   bash scripts/retrieval_infer.sh --query data/custom/query --dict data/custom/dict
#
# ============================================================================

set -e

# Default values
CONFIG="configs/retrieval/infer.yaml"
GPU="0"
QUERY_DIR=""
DICT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --query)
            QUERY_DIR="$2"
            shift 2
            ;;
        --dict)
            DICT_DIR="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="python src/retrieval/infer.py --config ${CONFIG} --gpu ${GPU}"

if [ -n "$QUERY_DIR" ]; then
    CMD="${CMD} --query_dir ${QUERY_DIR}"
fi

if [ -n "$DICT_DIR" ]; then
    CMD="${CMD} --dict_dir ${DICT_DIR}"
fi

# Print configuration
echo "=============================================="
echo "OBS Dictionary Retrieval"
echo "=============================================="
echo "Config: ${CONFIG}"
echo "GPU: ${GPU}"
if [ -n "$QUERY_DIR" ]; then
    echo "Query Dir: ${QUERY_DIR}"
fi
if [ -n "$DICT_DIR" ]; then
    echo "Dict Dir: ${DICT_DIR}"
fi
echo "=============================================="
echo ""

# Run inference
${CMD}

