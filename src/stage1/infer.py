"""
Stage 1: Font-Augmented Diffusion (FAD) Inference Script
Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval
"""

import argparse
import os
import yaml
import torch
import numpy as np
import torch.distributed as dist

from data.dataset import InferDataset
from models import DenoisingDiffusion, DiffusiveRestoration

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
    return os.path.dirname(os.path.dirname(current_dir))


def get_latest_checkpoint(base_path):
    """Find the checkpoint with the largest epoch number.
    
    Args:
        base_path: Base path for checkpoints (e.g., outputs/stage1/checkpoints/diffusion_model).
                   Checkpoints are expected to be named {base_path}_{epoch}.pth.tar
        
    Returns:
        Path to the latest checkpoint (without .pth.tar extension).
        
    Raises:
        FileNotFoundError: If no valid checkpoint is found.
    """
    checkpoint_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Find all matching checkpoint files
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(base_name + "_") and filename.endswith(".pth.tar"):
            try:
                # Extract epoch number from filename like "diffusion_model_299.pth.tar"
                epoch_str = filename[len(base_name) + 1:-8]  # Remove prefix and .pth.tar
                epoch = int(epoch_str)
                checkpoints.append((epoch, filename))
            except ValueError:
                continue
    
    if not checkpoints:
        # Fall back to the latest checkpoint without epoch number
        latest_path = base_path + ".pth.tar"
        if os.path.exists(latest_path):
            print(f"=> Using latest checkpoint: {os.path.basename(latest_path)}")
            return base_path
        raise FileNotFoundError(f"No valid checkpoints found matching: {base_path}_*.pth.tar")
    
    # Sort by epoch and get the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_filename = checkpoints[0]
    latest_path = os.path.join(checkpoint_dir, latest_filename[:-8])  # Remove .pth.tar
    
    print(f"=> Auto-selected checkpoint: {latest_filename} (epoch {latest_epoch})")
    
    return latest_path


def resolve_paths(config, project_root):
    """Convert relative paths in config to absolute paths."""
    config.data.train_data_dir = os.path.join(project_root, config.data.train_data_dir)
    config.data.test_data_dir = os.path.join(project_root, config.data.test_data_dir)
    config.data.test_save_dir = os.path.join(project_root, config.data.test_save_dir)
    config.data.val_save_dir = os.path.join(project_root, config.data.val_save_dir)
    config.data.tensorboard = os.path.join(project_root, config.data.tensorboard)
    
    # Handle checkpoint path: support both explicit path and auto-selection
    if config.training.resume == "auto":
        # Auto-select the latest checkpoint
        resume_base = os.path.join(project_root, config.training.resume_base)
        config.training.resume = get_latest_checkpoint(resume_base)
    else:
        # Use the explicitly specified checkpoint path
        config.training.resume = os.path.join(project_root, config.training.resume)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Stage 1 FAD Inference")
    parser.add_argument(
        "--config", 
        default='configs/stage1/infer.yaml', 
        type=str,
        help="Path to the config file (relative to project root)"
    )
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--seed', type=int, default=None, help="Random seed for inference")
    args = parser.parse_args()
    
    # Get project root and resolve config path
    project_root = get_project_root()
    config_path = os.path.join(project_root, args.config)
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    config = resolve_paths(config, project_root)
    
    # Setup distributed inference
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
    print(f"=> Using device: {device}")
    
    # Set random seed
    seed = args.seed if args.seed is not None else config.training.seed
    print(f"=> Using seed: {seed}")
    config.training.seed = seed
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Load dataset
    dataset = InferDataset(config)
    val_loader = dataset.get_loaders(parse_patches=False)

    # Create model
    print("=> Creating diffusion model for inference")
    diffusion = DenoisingDiffusion(config, test=True)
    model = DiffusiveRestoration(diffusion, config)

    # Create output directory
    os.makedirs(config.data.test_save_dir, exist_ok=True)
    print(f"=> Saving results to: {config.data.test_save_dir}")

    # Run inference
    model.restore(val_loader, r=config.data.grid_r)
    
    # Wait for all processes to complete
    dist.barrier()
    
    # Clean up GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if dist.get_rank() == 0:
        print("=> Inference completed successfully")


if __name__ == '__main__':
    main()

