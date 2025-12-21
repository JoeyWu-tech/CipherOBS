# CipherOBS: Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval

> **Official PyTorch implementation of the paper:**  
> *Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval*  
> Submitted to *Nature Machine Intelligence*

<p align="center">
  <a href="https://www.nature.com/natmachintell/"><img src="https://img.shields.io/badge/Nature-Machine%20Intelligence-red.svg" alt="Journal"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

---

## Abstract

Understanding humanity's earliest writing systems is crucial for reconstructing civilization's origins, yet many ancient scripts remain undeciphered. **CipherOBS** reframes the decipherment of Oracle Bone Script (OBS) from a closed-set classification problem to a **generative dictionary-based retrieval** task.

By synthesizing a comprehensive dictionary of plausible OBS variants for modern Chinese characters (using Font-Augmented Diffusion and Stroke Refinement), our system allows scholars to query unknown inscriptions and retrieve visually similar candidates with transparent evidence. This approach achieves state-of-the-art performance on unseen characters and remains robust against archaeological degradation.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pre-trained Weights](#pre-trained-weights)
- [Reproducing Results](#reproducing-results)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contact](#contact)

---

## System Architecture

CipherOBS employs a two-stage generative pipeline followed by dictionary retrieval:

<p align="center">
  <img src="assets/system_overview.png" alt="System Overview" width="800"/>
</p>

| Stage | Method | Description |
|-------|--------|-------------|
| **Stage 1** | Font-Augmented Diffusion (FAD) | Generates initial OBS drafts from modern Chinese characters |
| **Stage 2** | Stroke Refinement (SR) | Refines drafts using IDS-guided diffusion for structural fidelity |
| **Retrieval** | ConvNeXt Encoder | Matches query inscriptions against the synthetic dictionary |

---

## Quick Start

To reproduce our main results with minimal setup:

```bash
# Clone and install
git clone https://github.com/your-username/CipherOBS.git
cd CipherOBS && pip install -r requirements.txt

# Option 1: Run demo notebook (requires pre-computed features)
jupyter notebook notebooks/retrieval_demo.ipynb

# Option 2: Run retrieval on pre-generated dictionary
bash scripts/retrieval_infer.sh
```

For detailed reproduction options, see [Reproducing Results](#reproducing-results).

---

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (for GPU acceleration)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/CipherOBS.git
cd CipherOBS

# Create and activate conda environment
conda create -n cipherobs python=3.10 -y
conda activate cipherobs

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## Data Preparation

Prepare your data according to the following structure:

```
data/
├── train/
│   ├── input/                 # Stage 1: Source font images (modern Chinese)
│   ├── target/                # Stage 1: Target OBS images
│   ├── run_1/                 # Stage 2: Stage 1 outputs (content images)
│   └── TargetImage/
│       └── style0/            # Stage 2: Target style OBS images
│
├── test/
│   ├── input/                 # Test source images (modern Chinese)
│   └── target/                # Query OBS images (for retrieval evaluation)
│
├── style_reference.png        # Stage 2: Global style reference image
├── han_ids.txt                # IDS (Ideographic Description Sequence) encodings
└── glyphs.json                # Glyph vocabulary for IDS encoder
```

> **Note:** Detailed data preparation instructions are available in [`data/README.md`](data/README.md).

---

## Pre-trained Weights

> **Coming Soon:** Pre-trained model weights and pre-computed features will be released upon paper acceptance.  
> In the meantime, you can train the models from scratch using the instructions below.

The released files will be organized as follows:

```
weights/
├── stage1/
│   └── diffusion_model.pth.tar    # Stage 1: FAD checkpoint
├── stage2/
│   └── total_model.pth            # Stage 2: SR checkpoint
└── retrieval/
    └── convnext_encoder.pth       # Retrieval encoder checkpoint

data/
├── features/                       # Pre-computed features (for demo notebook)
│   ├── query_features.npy         # Query image features
│   └── dict_features.npy          # Dictionary image features
└── dictionary/                     # Pre-generated OBS dictionary (for retrieval)
    └── *.png                       # Generated OBS images
```

---

## Reproducing Results

We provide **four ways** to reproduce our results, from quickest (using pre-computed features) to complete (training from scratch):

| Method | Time | Requirements | Description |
|--------|------|--------------|-------------|
| [**Option 1: Demo Notebook**](#option-1-demo-notebook-fastest) | ~1 min | Pre-computed features | Reproduce main results instantly |
| [**Option 2: Retrieval Only**](#option-2-retrieval-only) | ~5 min | Pre-generated dictionary | Run retrieval on provided dictionary |
| [**Option 3: Full Inference**](#option-3-full-inference-pipeline) | ~2 days | Pre-trained weights | Generate dictionary + retrieval |
| [**Option 4: Train from Scratch**](#option-4-train-from-scratch) | ~14 days | Training data | Complete reproduction |

### Option 1: Demo Notebook (Fastest)

The quickest way to reproduce our main experimental results using pre-computed features:

```bash
# Launch Jupyter and open the demo notebook
jupyter notebook notebooks/retrieval_demo.ipynb
```

This notebook:
- Loads pre-extracted features for query images and dictionary
- Performs nearest-neighbor retrieval
- Computes Top-K accuracy metrics
- Visualizes retrieval results

**Expected Results:**

| Top-1 | Top-10 | Top-20 | Top-50 | Top-100 |
|-------|--------|--------|--------|---------|
| 21.20 | 54.33  | 66.76  | 86.15  | 96.85   |

> **Required files:** `data/features/query_features.npy`, `data/features/dict_features.npy`

### Option 2: Retrieval Only

If you have a pre-generated OBS dictionary (from Stage 1 & 2), run retrieval directly:

```bash
# Basic usage with default paths
bash scripts/retrieval_infer.sh

# Custom query and dictionary directories
bash scripts/retrieval_infer.sh --query data/test/target --dict data/dictionary --gpu 0
```

**Outputs:** `results/retrieval/` containing:
- `retrieval_results.json`: Top-K matches for each query
- `metrics.json`: Accuracy metrics (Top-1, Top-10, Top-20, Top-50, Top-100)

> **Required:** Query images in `data/test/target/`, dictionary in `data/dictionary/`

### Option 3: Full Inference Pipeline

Generate the OBS dictionary from modern Chinese characters using pre-trained weights, then perform retrieval:

```bash
# Step 1: Stage 1 - Generate draft OBS images
bash scripts/stage1_infer.sh

# Step 2: Stage 2 - Refine with stroke enhancement  
bash scripts/stage2_infer.sh

# Step 3: Run retrieval on generated dictionary
bash scripts/retrieval_infer.sh --dict outputs/stage2/inference
```

> **Required:** Pre-trained weights in `weights/stage1/`, `weights/stage2/`, `weights/retrieval/`

### Option 4: Train from Scratch

For complete reproduction including model training:

```bash
# Step 1: Train Stage 1 (FAD) model
bash scripts/stage1_train.sh

# Step 2: Generate Stage 1 outputs for Stage 2 training data
bash scripts/stage1_infer.sh

# Step 3: Train Stage 2 (SR) model
bash scripts/stage2_train.sh

# Step 4: Run full inference pipeline (see Option 3)
bash scripts/stage1_infer.sh
bash scripts/stage2_infer.sh
bash scripts/retrieval_infer.sh --dict outputs/stage2/inference
```

> **Training time:** ~24h for Stage 1, ~12h for Stage 2 (on 8× A100 GPUs)

---

## Training

### Prerequisites

Before training, ensure you have:
1. Prepared training data (see [Data Preparation](#data-preparation))
2. Configured GPU settings in the shell scripts

### Stage 1: FAD Training

Font-Augmented Diffusion learns to generate OBS drafts from modern Chinese characters.

**Configure training parameters:**

```yaml
# configs/stage1/train.yaml
training:
  batch_size: 16
  n_epochs: 300
  resume: 'outputs/stage1/checkpoints/diffusion_model'
```

**Start training:**

```bash
# Default: 8 GPUs with distributed training
bash scripts/stage1_train.sh
```

**Monitor training:**

```bash
tensorboard --logdir outputs/stage1/logs
```

### Stage 2: SR Training

Stroke Refinement further improves the visual quality using IDS-guided diffusion.

**Prerequisites:**
- Stage 1 outputs in `data/train/run_1/`
- Style reference images in `data/train/TargetImage/style0/`

**Configure training parameters:**

```yaml
# configs/stage2/train.yaml
training:
  batch_size: 32
  max_train_steps: 50000
```

**Start training:**

```bash
# Uses Accelerate for distributed training
bash scripts/stage2_train.sh
```

---

## Inference

### Stage 1: FAD Inference

Generate draft OBS images from modern Chinese characters.

**Configure inference:**

```yaml
# configs/stage1/infer.yaml
data:
  test_data_dir: 'data/test/'
  test_save_dir: 'outputs/stage1/results/'

training:
  resume: 'auto'  # Automatically selects latest checkpoint
```

**Run inference:**

```bash
bash scripts/stage1_infer.sh

# Or with specific seed
bash scripts/stage1_infer.sh 42
```

**Outputs:** `outputs/stage1/results/`

### Stage 2: SR Inference

Refine Stage 1 outputs with stroke-level enhancement.

**Configure inference:**

```yaml
# configs/stage2/infer.yaml
data:
  input_dir: 'outputs/stage1/results/'  # Stage 1 outputs
  style_image_path: 'data/style_reference.png'

model:
  ckpt_dir: 'auto'  # Automatically selects latest checkpoint
```

**Run inference:**

```bash
bash scripts/stage2_infer.sh
```

**Outputs:** `outputs/stage2/inference/`

### Dictionary Retrieval

Match query OBS images against the synthetic dictionary.

**Configure retrieval:**

```yaml
# configs/retrieval/infer.yaml
query_dir: 'data/test/target/'           # Query images (unknown OBS)
dict_dir: 'outputs/stage2/inference/'    # Generated dictionary (Stage 2 output)
model_dir: 'weights/retrieval/convnext_tri/'
```

**Run retrieval:**

```bash
# Basic usage
bash scripts/retrieval_infer.sh

# With custom directories
bash scripts/retrieval_infer.sh --query data/custom/query --dict data/custom/dict --gpu 0
```

**Outputs:** `results/retrieval/`

---

## Project Structure

```
CipherOBS/
├── configs/                    # Configuration files (YAML)
│   ├── stage1/
│   │   ├── train.yaml         # Stage 1 training config
│   │   └── infer.yaml         # Stage 1 inference config
│   ├── stage2/
│   │   ├── train.yaml         # Stage 2 training config
│   │   └── infer.yaml         # Stage 2 inference config
│   └── retrieval/
│       └── infer.yaml         # Retrieval inference config
│
├── notebooks/                  # Jupyter notebooks for demos
│   └── retrieval_demo.ipynb   # Quick reproduction with pre-computed features
│
├── scripts/                    # Shell scripts for training/inference
│   ├── stage1_train.sh
│   ├── stage1_infer.sh
│   ├── stage2_train.sh
│   ├── stage2_infer.sh
│   └── retrieval_infer.sh
│
├── src/                        # Source code
│   ├── stage1/                # FAD implementation
│   │   ├── models/            # Diffusion model architectures
│   │   ├── data/              # Data loading utilities
│   │   ├── train.py           # Training script
│   │   └── infer.py           # Inference script
│   ├── stage2/                # SR implementation
│   │   ├── models/            # UNet and encoder architectures
│   │   ├── data/              # IDS-aware data loading
│   │   ├── train.py           # Training script
│   │   └── infer.py           # Inference script
│   └── retrieval/             # Retrieval implementation
│       ├── models/            # ConvNeXt encoder
│       └── infer.py           # Retrieval inference
│
├── data/                       # Data directory (user-provided)
│   ├── features/              # Pre-computed features (for demo notebook)
│   ├── dictionary/            # Pre-generated OBS dictionary
│   └── README.md              # Data preparation guide
│
├── weights/                    # Model checkpoints (coming soon)
│   └── README.md              # Download instructions
│
├── requirements.txt            # Python dependencies
│
└── outputs/                    # Training/inference outputs (auto-generated)
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{CipherOBS2025,
  title   = {Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval},
  author  = {Wu, Yin and Zhang, Gangjian and Chen, Jiayu and Xu, Chang and Luo, Yuyu and Tang, Nan and Xiong, Hui},
  journal = {Submitted to Nature Machine Intelligence},
  note    = {Under review},
  year    = {2025}
}
```

---

## Contact

For questions or issues regarding the code or paper, please:

1. Open an issue in this repository
2. Contact the author: **Yin WU** ([ywu450@connect.hkust-gz.edu.cn](mailto:ywu450@connect.hkust-gz.edu.cn))

---

<p align="center">
  <i>Bridging millennia through machine intelligence</i>
</p>
