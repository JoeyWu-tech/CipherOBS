"""Font dataset for Stage 2 stroke refinement training."""

import os
import re
import random
import time
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..models.ids_encoder import IDSEncoder


def get_nonorm_transform(resolution):
    """Get transform without normalization.
    
    Args:
        resolution: Target image resolution.
        
    Returns:
        Composed transform.
    """
    transform = transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor()
    ])
    return transform


def extract_chinese_characters(directory_path):
    """Extract Chinese characters from filenames in a directory.
    
    Args:
        directory_path: Path to directory containing image files.
        
    Returns:
        Dictionary mapping characters to file paths.
    """
    chinese_dict = {}
    
    for filename in os.listdir(directory_path):
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', filename)
        file_path = os.path.join(directory_path, filename)
        
        for char in chinese_chars:
            if char not in chinese_dict:
                chinese_dict[char] = []
            chinese_dict[char].append(file_path)
    
    return chinese_dict


class FontDataset(Dataset):
    """Dataset for font generation training.
    
    This dataset loads content images (Stage 1 outputs), style references,
    and target images for Stage 2 stroke refinement training.
    """
    
    def __init__(self, args, phase, transforms=None, scr=False):
        """Initialize the dataset.
        
        Args:
            args: Configuration namespace containing data paths.
            phase: Dataset phase ('train' or 'test').
            transforms: List of transforms [content, style, target].
            scr: Whether to load negative samples for SCR loss.
        """
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg
        
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        # Build character-to-path mappings
        self.input_dict = extract_chinese_characters(
            os.path.join(self.root, self.phase, "run_1")
        )
        self.target_dict = extract_chinese_characters(
            os.path.join(self.root, self.phase, "TargetImage", "style0")
        )
        self.chinese_list = list(self.input_dict.keys())
        
        # Initialize IDS encoder
        ids_path = args.ids_path
        glyph_path = args.glyph_path
        self.ids_encoder = IDSEncoder(ids_path, glyph_path, 32)
        
        # Style reference image path
        self.style_image_path = args.style_image_path
        
    def get_path(self):
        """Build paths for target images and style-to-image mappings."""
        self.target_images = []
        self.style_to_images = {}
        
        target_image_dir = os.path.join(self.root, self.phase, "TargetImage")
        
        for style in os.listdir(target_image_dir):
            images_related_style = []
            style_dir = os.path.join(target_image_dir, style)
            
            for img in os.listdir(style_dir):
                # Skip problematic characters
                if '\ue83b' in img:
                    continue
                    
                img_path = os.path.join(style_dir, img)
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        """Get a training sample.
        
        Args:
            index: Sample index.
            
        Returns:
            Dictionary containing sample data.
        """
        # Use time-based seed for randomness
        current_time_seed = int(time.time() * 10 + index)
        random.seed(current_time_seed)
        
        # Randomly select a character
        random_number = random.randint(0, len(self.input_dict) - 1)
        chinese = self.chinese_list[random_number]
        
        # Get paths for this character
        content_paths = self.input_dict[chinese]
        target_paths = self.target_dict[chinese]
        
        content_image_path = random.sample(content_paths, 1)[0]
        target_image_path = random.sample(target_paths, 1)[0]
        
        # Parse target image name
        target_image_name = target_image_path.split('/')[-1]
        style, content = 'style0', target_image_name.split('.')[0]
        
        # Load content image (Stage 1 output)
        content_image = Image.open(content_image_path).convert('RGB')
        
        # Load style reference image
        style_image = Image.open(self.style_image_path).convert("RGB")
        
        # Load target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)
        
        # Apply transforms
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)
        
        # Extract character for IDS encoding
        char_name = content.split('_')[1] if '_' in content else content
        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            "fantizi": char_name
        }
        
        # Load negative samples for SCR loss if needed
        if self.scr:
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = os.path.join(
                    self.root, "train", "TargetImage",
                    choose_style, f"{choose_style}+{content}.jpg"
                )
                choose_neg_names.append(choose_neg_name)
            
            # Load negative images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images
        
        return sample

    def __len__(self):
        """Return dataset length."""
        return len(self.target_images)

