# Copyright (c) CipherOBS Authors. All rights reserved.
"""
Utility functions for OBS retrieval.

Provides model loading and configuration utilities that match the legacy code
for checkpoint compatibility.
"""

import os
import torch
import torch.nn as nn
import yaml

from .models.encoder import two_view_net


def get_model_list(dirname, key):
    """Find the latest model checkpoint in directory (legacy compatible)."""
    if not os.path.exists(dirname):
        print(f'No directory: {dirname}')
        return None
    gen_models = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f
    ]
    if not gen_models:
        return None
    gen_models.sort()
    return gen_models[-1]


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_model(model_dir, device='cuda'):
    """Load trained retrieval model (legacy compatible).
    
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
    
    # Find checkpoint (legacy logic)
    last_model_name = os.path.basename(get_model_list(model_dir, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if epoch != 'last':
        epoch = int(epoch)
    
    # Build model with config parameters (matches original training code)
    num_classes = config.get('nclasses', 701)
    block = config.get('block', 2)
    resnet = config.get('resnet', False)
    
    print(f'Building model: nclasses={num_classes}, block={block}, resnet={resnet}')
    
    model = two_view_net(
        class_num=num_classes, 
        block=block,
        return_f=False,
        resnet=resnet
    )
    
    # Load weights
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth' % epoch
    else:
        save_filename = 'net_%s.pth' % epoch
    
    save_path = os.path.join(model_dir, save_filename)
    print(f'Loading model from: {save_path}')
    
    state_dict = torch.load(save_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Set to eval mode and move to device
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
    Uses global mapping like the legacy code.
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
