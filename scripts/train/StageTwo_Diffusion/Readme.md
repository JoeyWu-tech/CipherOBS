# Stage Two Diffusion Model Training and Evaluation

This README provides instructions for training and evaluating a diffusion model using distributed computing with PyTorch.

## Training

To train the diffusion model, execute the following command:

```bash
bash train_phase_1.sh
```


## Evaluation

To evaluate the trained diffusion model, execute the following command:

```bash
python eval.py
```

## Directory Structure

- **StageTwo_Diffusion/outputs**: This directory contains our trained model from the second stage of training.

## Script Descriptions

- **train.py**: This is the training script for the diffusion model. It serves as the entry point for starting the training process and invokes necessary modules for model training and data loading.
- **eval.py**: This is the evaluation script for the diffusion model. It is used after training to assess the model's performance, such as the quality of generated images and its performance on the test set.
