# Data Directory

This directory contains datasets and auxiliary files for training and evaluation.

## Overview

CipherOBS uses a two-stage generative pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CipherOBS Data Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Modern Font Image ──► [Stage 1: FAD] ──► Draft OBS ──► [Stage 2: SR] ──►  │
│         +                                     +              +              │
│    Target OBS                            IDS Encoding   Style Reference     │
│                                                              │              │
│                                                              ▼              │
│                                                      Refined OBS Image      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
data/
├── train/                          # Training data
│   ├── input/                      # [Stage 1] Source font images
│   ├── target/                     # [Stage 1] Target OBS images
│   ├── run_1/                      # [Stage 2] Stage 1 outputs (content images)
│   └── TargetImage/
│       └── style0/                 # [Stage 2] Target style OBS images
│
├── test/                           # Test/Evaluation data
│   ├── input/                      # [Stage 1] Test input images
│   └── target/                     # [Stage 1] Ground truth for evaluation
│
├── style_reference.png             # [Stage 2] Global style reference image
├── han_ids.txt                     # [Stage 2] IDS (Ideographic Description Sequence) encodings
└── glyphs.json                     # [Stage 2] Glyph vocabulary for IDS encoder
```

---

## Stage 1: Font-Augmented Diffusion (FAD)

### Data Requirements

| Directory | Description |
|-----------|-------------|
| `train/input/` | Source font images (modern Chinese characters) |
| `train/target/` | Target Oracle Bone Script images |
| `test/input/` | Test source images |
| `test/target/` | Ground truth OBS images for evaluation |


### Image Specifications

| Property | Requirement |
|----------|-------------|
| Format | PNG or JPG |
| Color | RGB (3 channels) |
| Resolution | Variable  |

---

## Stage 2: Stroke Refinement (SR)

Stage 2 refines the draft OBS images from Stage 1 using IDS-guided diffusion.

### Data Requirements

| File/Directory | Description |
|----------------|-------------|
| `train/run_1/` | **Stage 1 outputs** - Draft OBS images to be refined |
| `train/TargetImage/style0/` | Target style OBS images (ground truth) |
| `style_reference.png` | Global style reference image |
| `han_ids.txt` | IDS encoding file |
| `glyphs.json` | Glyph vocabulary |

