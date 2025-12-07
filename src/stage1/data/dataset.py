"""
Dataset classes for Stage 1: Font-Augmented Diffusion (FAD)
Supports both training and inference data loading.
"""

import os
import re
import random
import time

import numpy as np
import torch
import torch.utils.data
import torchvision
import PIL
from torch.utils.data.distributed import DistributedSampler


def extract_chinese_characters(directory_path):
    """
    Extract Chinese characters from filenames and build a mapping dictionary.
    
    Args:
        directory_path: Path to directory containing image files
        
    Returns:
        Dictionary mapping Chinese characters to list of file paths
    """
    chinese_dict = {}
    
    for filename in os.listdir(directory_path):
        # Extract Chinese characters using regex
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', filename)
        file_path = os.path.join(directory_path, filename)
        
        for char in chinese_chars:
            if char not in chinese_dict:
                chinese_dict[char] = []
            chinese_dict[char].append(file_path)
    
    return chinese_dict


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset class with common functionality."""
    
    def __init__(self, data_dir, patch_size, n, keep_image_size, transforms, parse_patches=True):
        super().__init__()
        self.dir = data_dir
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.keep_image_size = keep_image_size
        
        self.input_names = sorted(os.listdir(os.path.join(data_dir, 'input')))
        self.gt_names = sorted(os.listdir(os.path.join(data_dir, 'target')))
    
    @staticmethod
    def get_params(img, output_size, n):
        """Get random crop parameters."""
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0], [0], h, w
        
        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw
    
    @staticmethod
    def n_random_crops(img, x, y, h, w):
        """Perform n random crops on an image."""
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)
    
    def resize_image(self, img):
        """Resize image based on configuration."""
        if not self.keep_image_size:
            return img.resize((100, 100), PIL.Image.LANCZOS)
        
        wd_new, ht_new = img.size
        
        if wd_new < self.patch_size or ht_new < self.patch_size:
            ratio = max(self.patch_size / wd_new, self.patch_size / ht_new)
            wd_new = int(wd_new * ratio)
            ht_new = int(ht_new * ratio)
        
        # Ensure dimensions are multiples of 16
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        
        return img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
    
    def process_for_inference(self, input_img, gt_img):
        """Process images for inference (whole-image restoration)."""
        wd_new, ht_new = input_img.size
        
        # Limit maximum dimension to 1024
        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024
        
        # Ensure dimensions are multiples of 16
        wd_new = int(16 * np.ceil(wd_new / 16.0))
        ht_new = int(16 * np.ceil(ht_new / 16.0))
        
        input_img = input_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.LANCZOS)
        
        return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0)
    
    def __len__(self):
        return len(self.input_names)


class TrainDatasetCore(BaseDataset):
    """
    Training dataset with random character sampling.
    Randomly samples characters and their corresponding images for diverse training.
    """
    
    def __init__(self, data_dir, patch_size, n, keep_image_size, transforms, parse_patches=True):
        super().__init__(data_dir, patch_size, n, keep_image_size, transforms, parse_patches)
        
        # Build character-to-filepath mapping for random sampling
        self.input_dict = extract_chinese_characters(os.path.join(data_dir, 'input'))
        self.target_dict = extract_chinese_characters(os.path.join(data_dir, 'target'))
        self.chinese_list = list(self.input_dict.keys())
    
    def get_images(self, index):
        # Use time-based seed for random character selection
        current_time_seed = int(time.time() * 10 + index)
        random.seed(current_time_seed)
        
        # Randomly select a character
        random_number = random.randint(0, len(self.input_dict) - 1)
        chinese = self.chinese_list[random_number]
        
        # Randomly select input and target images for this character
        input_name = random.sample(self.input_dict[chinese], 1)[0].split('/')[-1]
        gt_name = random.sample(self.target_dict[chinese], 1)[0].split('/')[-1]
        
        img_id = re.split('/', input_name)[-1][:-4]
        
        # Load images
        input_img = PIL.Image.open(os.path.join(self.dir, 'input', input_name)).convert('RGB')
        gt_img = PIL.Image.open(os.path.join(self.dir, 'target', gt_name)).convert('RGB')
        
        # Resize images
        input_img = self.resize_image(input_img)
        gt_img = self.resize_image(gt_img)
        
        if self.parse_patches:
            # Random patch extraction for training
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_crops = self.n_random_crops(input_img, i, j, h, w)
            gt_crops = self.n_random_crops(gt_img, i, j, h, w)
            
            outputs = [
                torch.cat([self.transforms(input_crops[k]), self.transforms(gt_crops[k])], dim=0)
                for k in range(self.n)
            ]
            return torch.stack(outputs, dim=0), img_id
        else:
            return self.process_for_inference(input_img, gt_img), img_id
    
    def __getitem__(self, index):
        return self.get_images(index)


class InferDatasetCore(BaseDataset):
    """
    Inference dataset for sequential image processing.
    Processes images in order for reproducible inference.
    """
    
    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        
        # Load images
        input_img = PIL.Image.open(os.path.join(self.dir, 'input', input_name)).convert('RGB')
        gt_img = PIL.Image.open(os.path.join(self.dir, 'input', gt_name)).convert('RGB')
        
        # Resize images
        input_img = self.resize_image(input_img)
        gt_img = self.resize_image(gt_img)
        
        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_crops = self.n_random_crops(input_img, i, j, h, w)
            gt_crops = self.n_random_crops(gt_img, i, j, h, w)
            
            outputs = [
                torch.cat([self.transforms(input_crops[k]), self.transforms(gt_crops[k])], dim=0)
                for k in range(self.n)
            ]
            return torch.stack(outputs, dim=0), img_id
        else:
            return self.process_for_inference(input_img, gt_img), img_id
    
    def __getitem__(self, index):
        return self.get_images(index)


class TrainDataset:
    """Data loader wrapper for training."""
    
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def get_loaders(self, parse_patches=True):
        train_dataset = TrainDatasetCore(
            data_dir=self.config.data.train_data_dir,
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            keep_image_size=self.config.data.training_keep_image_size,
            transforms=self.transforms,
            parse_patches=parse_patches
        )
        
        val_dataset = TrainDatasetCore(
            data_dir=self.config.data.test_data_dir,
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            keep_image_size=self.config.data.testing_keep_image_size,
            transforms=self.transforms,
            parse_patches=parse_patches
        )
        
        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            sampler=DistributedSampler(train_dataset),
            num_workers=self.config.data.num_workers,
            prefetch_factor=2,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            sampler=DistributedSampler(val_dataset)
        )
        
        return train_loader, val_loader


class InferDataset:
    """Data loader wrapper for inference."""
    
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    
    def get_loaders(self, parse_patches=False):
        val_dataset = InferDatasetCore(
            data_dir=self.config.data.test_data_dir,
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            keep_image_size=self.config.data.testing_keep_image_size,
            transforms=self.transforms,
            parse_patches=parse_patches
        )
        
        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
            sampler=DistributedSampler(val_dataset)
        )
        
        return val_loader

