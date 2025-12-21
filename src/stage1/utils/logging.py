"""
Logging utilities for model checkpoints and image saving.
"""

import os
import torch
import torchvision.utils as tvu


def save_image(img, file_directory):
    """
    Save image tensor to file.
    
    Args:
        img: Image tensor
        file_directory: Path to save the image
    """
    os.makedirs(os.path.dirname(file_directory), exist_ok=True)
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state
        filename: Path to save the checkpoint (without extension)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    """
    Load model checkpoint.
    
    Args:
        path: Path to the checkpoint file
        device: Device to load the checkpoint to
        
    Returns:
        Loaded checkpoint dictionary
    """
    if device is None:
        return torch.load(path, weights_only=False)
    else:
        return torch.load(path, map_location=device, weights_only=False)

