# Stage One Diffusion Model Training and Evaluation

This README provides instructions for training and evaluating a diffusion model using distributed computing with PyTorch.

## Training

To train the diffusion model, execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=1234 train_diffusion.py
```

### Explanation:

- **CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7**: Specifies which GPUs to use for training.
- **python -m torch.distributed.run**: Runs the script in distributed mode.
- **--nproc_per_node=8**: Number of processes to run on each node (8 GPUs in this case).
- **--nnodes=1**: Number of nodes to use (1 node in this case).
- **--node_rank=0**: The rank of the node for multi-node training (0 for single-node).
- **--master_addr=localhost**: The address of the master node.
- **--master_port=1234**: The port on the master node.
- **train_diffusion.py**: The script to start the training process.

## Evaluation

To evaluate the trained diffusion model, execute the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=1234 eval_diffusion.py
```

### Explanation:

- The command structure is similar to the training command.
- **eval_diffusion.py**: The script used to evaluate the performance of the trained model.

## Directory Structure

- **StageOne_Diffusion/output**: This directory contains our trained model from the first stage of training.
- **configs.yml**: This file stores various hyperparameters used during training and evaluation.

## Script Descriptions

- **train_diffusion.py**: This is the training script for the diffusion model. It serves as the entry point for starting the training process and invokes necessary modules for model training and data loading.
- **eval_diffusion.py**: This is the evaluation script for the diffusion model. It is used after training to assess the model's performance, such as the quality of generated images and its performance on the test set.
