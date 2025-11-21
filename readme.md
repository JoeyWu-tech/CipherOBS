# Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval

**Official implementation of CipherOBS**


> **Note:** This repository is currently under construction to support the manuscript submitted to *Nature Machine Intelligence*. The full source code, pre-trained models, and data processing scripts will be made publicly available immediately upon acceptance/publication.

## 📝 Abstract

Understanding humanity's earliest writing systems is crucial for reconstructing civilization's origins, yet many ancient scripts remain undeciphered. **CipherOBS** reframes the decipherment of Oracle Bone Script (OBS) from a closed-set classification problem to a **generative dictionary-based retrieval** task.

By synthesizing a comprehensive dictionary of plausible OBS variants for modern Chinese characters (using Font-Augmented Diffusion and Stroke Refinement), our system allows scholars to query unknown inscriptions and retrieve visually similar candidates with transparent evidence. This approach achieves state-of-the-art performance on unseen characters and remains robust against archaeological degradation.

## 🚀 Upcoming Release Roadmap

We are preparing a comprehensive release to ensure full reproducibility of the results presented in the paper. The repository will be organized into the following modules:

### 1\. Source Code (`/src`)

  * **Generative Dictionary Synthesis:**
      * **Font-Augmented Diffusion (FAD):** Code for generating initial OBS drafts from modern characters.
      * **Stroke Refinement (SR):** IDS-guided diffusion refinement to enforce structural fidelity.
  * **Retrieval Engine:**
      * **ConvNeXt Encoder:** The metric learning backbone for embedding ancient scripts.
      * **Dictionary Indexing:** Scripts to build and search the synthetic vector database.

### 2\. Pre-trained Models (`/weights`)

We will provide model checkpoints to allow users to run inference without retraining:

  * `fad_diffusion.pth`: Weights for the Stage-1 generator.
  * `sr_refinement.pth`: Weights for the Stage-2 refinement model.
  * `retrieval_encoder.pth`: The trained visual encoder for the retrieval system.

### 3\. Data & Reproduction (`/data` & `/scripts`)

  * **Datasets:** Preprocessing scripts for the **HUST-OBS** and **EVOBC** benchmarks.
  * **Reproduction Scripts:** One-click shell scripts to reproduce the Top-N accuracy results reported in Table 1 of the manuscript.
  * **Human Evaluation:** The interface code used for the expert-in-the-loop user study.

## 📂 Repository Structure Preview

The final repository will follow this structure:

```text
CipherOBS/
├── configs/                 # Configuration files for FAD, SR, and Retrieval
├── data/                    # Data preprocessing scripts (HUST-OBS / EVOBC)
├── notebooks/               # Jupyter demos for zero-shot decipherment
├── scripts/                 # Shell scripts for training and evaluation
├── src/                     # Core source code
│   ├── models/              # Diffusion U-Net and ConvNeXt definitions
│   ├── pipeline/            # Dictionary construction and search logic
│   └── trainer/             # Training loops
└── README.md                # Documentation
```

## 📅 Citation

If you find this work useful, please consider citing our paper (BibTeX will be updated upon publication):

```bibtex
@article{CipherOBS2025,
  title={Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval},
  author={Wu, Yin and Zhang, Gangjian and Chen, Jiayu and Xu, Chang and Luo, Yuyu and Tang, Nan and Xiong, Hui},
  journal={Submitted to Nature Machine Intelligence},
  year={2025}
}
```

## ✉️ Contact

For questions regarding the code or the paper, please open an issue in this repository or contact the corresponding author:

  * **Yin WU**: `ywu450@connect.hkust-gz.edu.cn`