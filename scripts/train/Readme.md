# Project Repository

This repository contains code and data for a project utilizing diffusion models. Below is a detailed breakdown of each folder's role within the project.

## Folder Structure
```

```
Project-Root/
│
├── diff-han/
│   └── [Files related to Chinese character IDS sequence]
│
├── new_data/
│   ├── [Training data files]
│   └── [Testing data files]
│
├── StageOne_Diffusion/
│   └── [Source code files for the first-stage diffusion model]
│
└── StageTwo_Diffusion/
    └── [Source code files for the second-stage diffusion model]
```

```
### Descriptions 

- **diff-han**: This directory includes resources related to the structural analysis of Chinese characters through the Ideographic Description Sequence (IDS) comparison table.

- **new_data**: Contains datasets used for training and testing purposes within the project.

- **StageOne_Diffusion**: Consists of the source code required to implement the first-stage diffusion model.

- **StageTwo_Diffusion**: Consists of the source code required to implement the second-stage diffusion model.


### Environment Configuration 

conda create -n  cypherOBS python=3.10
conda activate cypherOBS
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
