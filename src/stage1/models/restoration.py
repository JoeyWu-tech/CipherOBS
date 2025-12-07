"""
Diffusive Restoration Module for Stage 1 FAD
Implements image restoration using pre-trained diffusion models.
"""

import os
import torch
from tqdm import tqdm

from ..utils import logging


def data_transform(X):
    """Transform data to [-1, 1] range."""
    return 2 * X - 1.0


def inverse_data_transform(X):
    """Transform data back to [0, 1] range."""
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    """
    Image restoration using diffusion models.
    
    Uses a pre-trained diffusion model to restore images through
    the reverse diffusion process with overlapping patches.
    """
    
    def __init__(self, diffusion, config):
        super().__init__()
        self.config = config
        self.diffusion = diffusion

        # Load pretrained model
        pretrained_model_path = self.config.training.resume + '.pth.tar'
        assert os.path.isfile(pretrained_model_path), f'Pretrained model not found: {pretrained_model_path}'
        
        self.diffusion.load_ddm_ckpt(pretrained_model_path, ema=True)
        self.diffusion.model.eval()
        self.diffusion.model.requires_grad_(False)

    def restore(self, val_loader, r=None):
        """
        Restore images from the validation loader.
        
        Args:
            val_loader: DataLoader with input images
            r: Grid step size for overlapping patches
        """
        image_folder = self.config.data.test_save_dir
        
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_loader), desc="Restoring images"):
                print(f"=> Processing image: {y}")
                
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_output = self.diffusive_restoration(x_cond, r=r)
                x_output = inverse_data_transform(x_output)
                
                logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))

    def diffusive_restoration(self, x_cond, r=None):
        """
        Perform diffusive restoration with overlapping patches.
        
        Args:
            x_cond: Conditional input image
            r: Grid step size (default: 8)
            
        Returns:
            Restored image tensor
        """
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_output = self.diffusion.sample_image(x_cond, x, patch_locs=corners, patch_size=p_size)
        
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        """
        Calculate grid indices for overlapping patch processing.
        
        Args:
            x_cond: Input image tensor
            output_size: Patch size
            r: Step size between patches (default: 8)
            
        Returns:
            Lists of height and width indices
        """
        _, c, h, w = x_cond.shape
        r = 8 if r is None else r
        
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        
        return h_list, w_list

    def web_restore(self, image, r=None):
        """
        Restore a single image (for web interface).
        
        Args:
            image: Input image tensor
            r: Grid step size
            
        Returns:
            Restored image tensor
        """
        with torch.no_grad():
            image_cond = image.to(self.diffusion.device)
            image_output = self.diffusive_restoration(image_cond, r=r)
            image_output = inverse_data_transform(image_output)
            return image_output

