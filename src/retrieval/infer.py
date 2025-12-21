#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) CipherOBS Authors. All rights reserved.
"""
Dictionary-based OBS Retrieval Inference.

This script performs Oracle Bone Script decipherment via dictionary retrieval.
Given query OBS images and a generated dictionary, it:
1. Extracts visual features using a ConvNeXt-based encoder
2. Computes cosine similarity between query and dictionary features
3. Applies voting-based reranking for improved accuracy
4. Outputs Top-N retrieval results

Usage:
    python infer.py --config configs/retrieval/infer.yaml

Reference:
    Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval
"""

import argparse
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

from retrieval.transforms import get_query_transforms, get_dictionary_transforms
from retrieval.utils import load_model, parse_filename, LabelMapper


# ============================================================================
# Dataset
# ============================================================================

class RetrievalDataset(Dataset):
    """Dataset for OBS retrieval.
    
    Loads images from a directory and extracts labels from filenames.
    """
    
    def __init__(self, data_dir: str, transform, label_mapper: LabelMapper):
        self.data_dir = data_dir
        self.transform = transform
        self.label_mapper = label_mapper
        
        self.samples = []  # List of (filepath, label_id)
        self.imgs = []     # For compatibility with legacy code
        
        for filename in os.listdir(data_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                continue
            
            label = parse_filename(filename)
            if label is None:
                continue
            
            filepath = os.path.join(data_dir, filename)
            label_id = label_mapper.get_id(label)
            self.samples.append((filepath, label_id))
            self.imgs.append((filepath, label_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label_id = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_id


# ============================================================================
# Feature Extraction (matches legacy test.py exactly)
# ============================================================================

def fliplr(img):
    """Flip horizontal (legacy compatible)."""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    view_index: int = 1,
    device: str = 'cuda',
    scales: List[float] = [1.0]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from images (matches legacy test.py exactly).
    
    Args:
        model: Feature encoder model.
        dataloader: DataLoader for images.
        view_index: 1 for gallery/dictionary, 3 for query (legacy convention).
        device: Device to run inference on.
        scales: List of scale factors for multi-scale extraction.
        
    Returns:
        features: Tensor of shape (N, D) with L2-normalized features.
        labels: Tensor of shape (N,) with label IDs.
    """
    features = torch.FloatTensor()
    labels = torch.FloatTensor()
    
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=f'Extracting features (view={view_index})'):
            img, label = data
            n, c, h, w = img.size()
            
            ff = None
            for i in range(2):  # 0=normal, 1=flipped
                if i == 1:
                    img = fliplr(img)
                input_img = Variable(img.to(device))
                
                for scale in scales:
                    if scale != 1:
                        input_img = nn.functional.interpolate(
                            input_img, scale_factor=scale, 
                            mode='bilinear', align_corners=False
                        )
                    
                    # View index: 1=gallery (first output), 3=query (second output)
                    if view_index == 1:
                        outputs, _ = model(input_img, None)
                    elif view_index == 3:
                        _, outputs = model(None, input_img)
                    else:
                        raise ValueError(f"Invalid view_index: {view_index}")
                    
                    if ff is None:
                        ff = outputs
                    else:
                        ff = ff + outputs
            
            # Norm feature (matches legacy code exactly)
            if len(ff.shape) == 3:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(ff.size(-1))
                ff = ff.div(fnorm.expand_as(ff))
                ff = ff.view(ff.size(0), -1)
            else:
                fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                ff = ff.div(fnorm.expand_as(ff))
            
            features = torch.cat((features, ff.data.cpu()), 0)
            labels = torch.cat((labels, label.float()), 0)
    
    return features, labels


# ============================================================================
# Voting-based Reranking (matches legacy test.py exactly)
# ============================================================================

def voting_rerank(
    qf: torch.Tensor,
    ql: int,
    gf: torch.Tensor,
    gl: np.ndarray
) -> List[int]:
    """Voting-based reranking for a single query (matches legacy exactly).
    
    Args:
        qf: Query feature vector (D,).
        ql: Query label (int).
        gf: Gallery features (N, D).
        gl: Gallery labels (N,).
        
    Returns:
        new_rank: Binary list where position i is 1 if query's true label
                  appears in top-(i+1) after voting rerank.
    """
    if not isinstance(gl, np.ndarray):
        gl = np.array(gl)
    
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    
    # Sort descending
    index = np.argsort(score)[::-1]
    
    # Find good and junk indices
    query_index = np.argwhere(gl == ql)
    good_index = query_index
    junk_index = np.argwhere(gl == -1)
    
    if good_index.size == 0:
        new_rank = [0] * len(index)
        return new_rank
    
    # Remove junk_index (legacy behavior)
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    
    # Find good_index positions
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()
    
    if len(rows_good) == 0:
        new_rank = [0] * len(index)
        return new_rank
    
    # Core voting reranking algorithm
    calculated_label = []
    label_score = {}
    for i in range(rows_good[0] + 1):
        current_label = gl[index[i]]
        if current_label not in calculated_label:
            calculated_label.append(current_label)
            label_index = np.argwhere(gl == current_label)
            label_mask = np.in1d(index, label_index)
            rows_good_label = np.argwhere(label_mask == True)
            rows_good_label = rows_good_label.flatten()
            label_score[current_label] = sum(rows_good_label) / len(rows_good_label)
    
    sorted_labels = [k for k, v in sorted(label_score.items(), key=lambda x: x[1])]
    new_ql_rank = len(sorted_labels)
    for i in range(len(sorted_labels)):
        if sorted_labels[i] == ql:
            new_ql_rank = i
            break
    
    new_rank = [0] * len(index)
    for i in range(len(index)):
        if i >= new_ql_rank:
            new_rank[i] = 1
    
    return new_rank


def compute_voting_rerank_results(
    query_feature: torch.Tensor,
    query_label: List[int],
    gallery_feature: torch.Tensor,
    gallery_label: List[int],
    device: str = 'cuda'
) -> List[float]:
    """Compute voting rerank results for all queries (matches legacy)."""
    print("Performing voting-based reranking...")
    
    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)
    gallery_label_np = np.array(gallery_label)
    
    new_rank_all = [0] * len(gallery_label)
    
    for i in tqdm(range(len(query_label)), desc="Voting rerank"):
        new_rank_tmp = voting_rerank(
            query_feature[i], 
            query_label[i], 
            gallery_feature, 
            gallery_label_np
        )
        new_rank_all = [x + y for x, y in zip(new_rank_all, new_rank_tmp)]
    
    # Average
    new_rank_all = [x / len(query_label) for x in new_rank_all]
    
    return new_rank_all


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='OBS Dictionary Retrieval Inference')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--query_dir', type=str, default=None,
                        help='Override query image directory')
    parser.add_argument('--dict_dir', type=str, default=None,
                        help='Override dictionary image directory')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Override paths if provided
    if args.query_dir:
        config['query_dir'] = args.query_dir
    if args.dict_dir:
        config['dict_dir'] = args.dict_dir
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model_dir = config['model_dir']
    model, model_config, epoch = load_model(model_dir, device)
    print(f"Loaded model from epoch: {epoch}")
    
    # Setup transforms with config from model
    h = model_config.get('h', config.get('height', 256))
    w = model_config.get('w', config.get('width', 256))
    # Note: Use pad from inference config (not model config) to match legacy test.py behavior
    # Legacy test.py uses argparse default pad=0, does not load from model config
    pad = config.get('pad', 0)
    
    # Multi-scale (parse from config, default to 1)
    ms_str = config.get('ms', '1')
    if isinstance(ms_str, str):
        scales = [math.sqrt(float(s)) for s in ms_str.split(',')]
    elif isinstance(ms_str, list):
        scales = [math.sqrt(s) for s in ms_str]
    else:
        scales = [1.0]
    
    print(f"Image size: {h}x{w}, pad: {pad}, scales: {scales}")
    
    query_transform = get_query_transforms(h, w, pad)
    dict_transform = get_dictionary_transforms(h, w)
    
    # Setup label mapper
    label_mapper = LabelMapper()
    label_mapper.scan_directories(config['query_dir'], config['dict_dir'])
    
    # Create datasets
    query_dataset = RetrievalDataset(config['query_dir'], query_transform, label_mapper)
    dict_dataset = RetrievalDataset(config['dict_dir'], dict_transform, label_mapper)
    
    print(f"Query images: {len(query_dataset)}")
    print(f"Dictionary images: {len(dict_dataset)}")
    
    batch_size = config.get('batch_size', 8)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dict_loader = DataLoader(dict_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Extract features using legacy view indices
    # Gallery/Dictionary = view 1, Query = view 3
    start_time = time.time()
    
    gallery_feature, gallery_labels = extract_features(
        model, dict_loader, view_index=1, device=device, scales=scales
    )
    query_feature, query_labels = extract_features(
        model, query_loader, view_index=3, device=device, scales=scales
    )
    
    extraction_time = time.time() - start_time
    print(f"Feature extraction completed in {extraction_time:.1f}s")
    
    # Get labels as lists (matches legacy)
    gallery_label = [int(l) for l in gallery_labels.numpy()]
    query_label = [int(l) for l in query_labels.numpy()]
    
    # Compute voting rerank results
    new_rank_results = compute_voting_rerank_results(
        query_feature, query_label,
        gallery_feature, gallery_label,
        device
    )
    
    # Print results
    num_gallery = len(gallery_label)
    print('\nTop1:%.2f Top10:%.2f Top20:%.2f Top50:%.2f Top100:%.2f' % (
        new_rank_results[0] * 100,
        new_rank_results[min(9, num_gallery - 1)] * 100,
        new_rank_results[min(19, num_gallery - 1)] * 100,
        new_rank_results[min(49, num_gallery - 1)] * 100,
        new_rank_results[min(99, num_gallery - 1)] * 100,
    ))
    
    # Save results
    output_dir = config.get('output_dir', './results/retrieval')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'retrieval_results.txt')
    with open(output_file, 'w') as f:
        f.write("OBS Dictionary Retrieval Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Query directory: {config['query_dir']}\n")
        f.write(f"Dictionary directory: {config['dict_dir']}\n")
        f.write(f"Number of queries: {len(query_dataset)}\n")
        f.write(f"Dictionary size: {len(dict_dataset)}\n")
        f.write("=" * 50 + "\n")
        f.write('Top1:%.2f Top10:%.2f Top20:%.2f Top50:%.2f Top100:%.2f\n' % (
            new_rank_results[0] * 100,
            new_rank_results[min(9, num_gallery - 1)] * 100,
            new_rank_results[min(19, num_gallery - 1)] * 100,
            new_rank_results[min(49, num_gallery - 1)] * 100,
            new_rank_results[min(99, num_gallery - 1)] * 100,
        ))
    
    print(f"\nResults saved to: {output_file}")
    
    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.1f}s")


if __name__ == '__main__':
    main()
