# Copyright (c) CipherOBS Authors. All rights reserved.
"""
Utility functions for OBS retrieval.

Provides model loading and configuration utilities.
"""

import os
import torch
import yaml

# Handle both package and direct imports
try:
    from .models.encoder import two_view_net
except ImportError:
    from models.encoder import two_view_net


def get_model_path(model_dir, pattern='net'):
    """Find the latest model checkpoint in directory.
    
    Args:
        model_dir: Directory containing model checkpoints.
        pattern: Filename pattern to match.
        
    Returns:
        Path to the latest checkpoint, or None if not found.
    """
    if not os.path.exists(model_dir):
        print(f'Model directory not found: {model_dir}')
        return None
    
    model_files = [
        os.path.join(model_dir, f) for f in os.listdir(model_dir)
        if os.path.isfile(os.path.join(model_dir, f)) and pattern in f and f.endswith('.pth')
    ]
    
    if not model_files:
        return None
    
    model_files.sort()
    return model_files[-1]


def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_model(model_dir, device='cuda'):
    """Load trained retrieval model.
    
    The model weights should be placed in the specified model_dir with:
    - opts.yaml: Model configuration
    - net_*.pth: Model checkpoint
    
    Args:
        model_dir: Directory containing model checkpoint and config.
        device: Device to load model on.
        
    Returns:
        Tuple of (model, config, epoch).
    """
    # Load config
    config_path = os.path.join(model_dir, 'opts.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Find checkpoint
    checkpoint_path = get_model_path(model_dir, 'net')
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in: {model_dir}")
    
    print(f'Loading model from: {checkpoint_path}')
    
    # Extract epoch from filename
    epoch = os.path.basename(checkpoint_path).split('_')[1].split('.')[0]
    
    # Build model with config parameters (matches original training code)
    num_classes = config.get('nclasses', 701)
    block = config.get('block', 2)
    resnet = config.get('resnet', False)
    
    model = two_view_net(
        class_num=num_classes, 
        block=block,
        return_f=False,
        resnet=resnet
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set to eval mode
    model = model.to(device)
    model.eval()
    
    # Add config values for reference
    config['height'] = config.get('h', 256)
    config['width'] = config.get('w', 256)
    
    return model, config, epoch


def parse_filename(filename):
    """Parse label from filename.
    
    Expected format: test_{label}_{index}.png or test_{label}_{index}_output.png
    
    Args:
        filename: Image filename.
        
    Returns:
        Label string, or None if parsing fails.
    """
    if not filename.startswith('test_'):
        return None
    
    parts = filename.split('_')
    if len(parts) >= 3:
        return parts[1]
    
    return None


class LabelMapper:
    """Maps string labels to numeric IDs and vice versa.
    
    Ensures consistent label encoding between query and dictionary datasets.
    """
    
    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = {}
        self._next_id = 0
    
    def get_id(self, label):
        """Get numeric ID for a string label."""
        if label not in self.label_to_id:
            self.label_to_id[label] = self._next_id
            self.id_to_label[self._next_id] = label
            self._next_id += 1
        return self.label_to_id[label]
    
    def get_label(self, id):
        """Get string label for a numeric ID."""
        return self.id_to_label.get(id, None)
    
    def scan_directories(self, *directories):
        """Pre-scan directories to build complete label mapping.
        
        Args:
            *directories: Directories to scan for image files.
        """
        for data_dir in directories:
            if not os.path.exists(data_dir):
                continue
            
            for filename in os.listdir(data_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                    continue
                
                label = parse_filename(filename)
                if label is not None:
                    self.get_id(label)
        
        print(f"Found {len(self.label_to_id)} unique labels")
    
    def __len__(self):
        return len(self.label_to_id)
