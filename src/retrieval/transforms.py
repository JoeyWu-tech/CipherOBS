# Copyright (c) CipherOBS Authors. All rights reserved.
"""
Image transforms for OBS retrieval.

Provides preprocessing transforms for query and dictionary images with
multi-scale and flip augmentation during feature extraction.
"""

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_query_transforms(height=256, width=256, pad=0):
    """Get transforms for query OBS images.
    
    Args:
        height: Target image height.
        width: Target image width.
        pad: Padding size for query images.
        
    Returns:
        Composed transforms.
    """
    transform_list = [
        transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
    ]
    
    if pad > 0:
        transform_list.append(QueryPadTransform(pad=pad, size=width))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def get_dictionary_transforms(height=256, width=256):
    """Get transforms for dictionary images.
    
    Args:
        height: Target image height.
        width: Target image width.
        
    Returns:
        Composed transforms.
    """
    return transforms.Compose([
        transforms.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class QueryPadTransform:
    """Pad query images to handle edge effects.
    
    Adds zero-padding to the left side of the image, which can help
    with boundary artifacts in OBS rubbings.
    
    Args:
        pad: Number of pixels to pad.
        size: Target width after padding.
    """
    
    def __init__(self, pad=20, size=256):
        self.pad = pad
        self.size = size
    
    def __call__(self, img):
        if self.pad <= 0:
            return img
        
        img_array = np.array(img).copy()
        img_part = img_array[:, 0:self.pad, :]
        img_pad = np.zeros_like(img_part, dtype=np.uint8)
        image = np.concatenate((img_pad, img_array), axis=1)
        image = image[:, 0:self.size, :]
        
        return Image.fromarray(image.astype('uint8')).convert('RGB')


