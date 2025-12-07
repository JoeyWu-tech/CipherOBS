"""
Stage 1: Font-Augmented Diffusion (FAD) Training Script
Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval
"""

import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np
import torch.distributed as dist

from data.dataset import TrainDataset
from models import DenoisingDiffusion
from utils import misc

os.environ['NCCL_P2P_DISABLE'] = '1'


def dict2namespace(config):
    """Convert dictionary to namespace for attribute-style access."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from src/stage1 to project root
    return os.path.dirname(os.path.dirname(current_dir))


def resolve_paths(config, project_root):
    """Convert relative paths in config to absolute paths."""
    # Data paths
    config.data.train_data_dir = os.path.join(project_root, config.data.train_data_dir)
    config.data.test_data_dir = os.path.join(project_root, config.data.test_data_dir)
    config.data.val_save_dir = os.path.join(project_root, config.data.val_save_dir)
    config.data.tensorboard = os.path.join(project_root, config.data.tensorboard)
    
    # Model checkpoint path
    config.training.resume = os.path.join(project_root, config.training.resume)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Stage 1 FAD Training")
    parser.add_argument(
        "--config", 
        default='configs/stage1/train.yaml', 
        type=str,
        help="Path to the config file (relative to project root)"
    )
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()
    
    # Get project root and resolve config path
    project_root = get_project_root()
    config_path = os.path.join(project_root, args.config)
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config = resolve_paths(config, project_root)
    
    # Setup distributed training
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.dist_url = 'env://'
    
    print(f'| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}', flush=True)
    
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend='nccl', 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    dist.barrier()
    
    # Setup device
    config.local_rank = args.gpu
    device = torch.device("cuda", config.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    config.device = device

    # Scale learning rate based on effective batch size
    eff_batch_size = config.training.batch_size * misc.get_world_size()
    assert config.optim.lr is not None
    config.optim.lr = config.optim.lr * eff_batch_size / 8

    # Set random seeds for reproducibility
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True

    # Load dataset
    dataset = TrainDataset(config)
    _, val_loader = dataset.get_loaders()

    # Create and train model
    if dist.get_rank() == 0:
        print("=> Creating denoising diffusion model", flush=True)
        print(f"   Learning rate: {config.optim.lr:.2e}")
        print(f"   Effective batch size: {eff_batch_size}")
    
    diffusion = DenoisingDiffusion(config)
    diffusion.train(dataset)


if __name__ == "__main__":
    main()

