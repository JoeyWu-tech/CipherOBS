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
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

# Handle both package and direct imports
try:
    from .transforms import get_query_transforms, get_dictionary_transforms
    from .utils import load_model, parse_filename, LabelMapper
except ImportError:
    from transforms import get_query_transforms, get_dictionary_transforms
    from utils import load_model, parse_filename, LabelMapper

# Note: Feature extraction model should be placed in weights/ConvNext/
# See weights/ConvNext/README.md for download instructions


# ============================================================================
# Dataset
# ============================================================================

class RetrievalDataset(Dataset):
    """Dataset for OBS retrieval.
    
    Loads images from a directory and extracts labels from filenames.
    
    Args:
        data_dir: Directory containing images.
        transform: Image transforms to apply.
        label_mapper: LabelMapper instance for consistent label encoding.
    """
    
    def __init__(self, data_dir: str, transform, label_mapper: LabelMapper):
        self.data_dir = data_dir
        self.transform = transform
        self.label_mapper = label_mapper
        
        self.samples = []  # List of (filepath, label_id)
        
        for filename in sorted(os.listdir(data_dir)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                continue
            
            label = parse_filename(filename)
            if label is None:
                continue
            
            filepath = os.path.join(data_dir, filename)
            label_id = label_mapper.get_id(label)
            self.samples.append((filepath, label_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label_id = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_id


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    view_type: str = 'query',
    device: str = 'cuda',
    scales: List[float] = [1.0]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features from images with multi-scale and flip augmentation.
    
    Args:
        model: Feature encoder model.
        dataloader: DataLoader for images.
        view_type: 'query' or 'dictionary'.
        device: Device to run inference on.
        scales: List of scale factors for multi-scale extraction.
        
    Returns:
        features: Tensor of shape (N, D) with L2-normalized features.
        labels: Tensor of shape (N,) with label IDs.
    """
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for images, batch_labels in tqdm(dataloader, desc=f'Extracting {view_type} features'):
            images = images.to(device)
            batch_size = images.size(0)
            
            # Multi-scale and flip augmentation
            batch_features = None
            for scale in scales:
                for flip in [False, True]:
                    # Apply flip
                    if flip:
                        imgs = torch.flip(images, dims=[3])  # Horizontal flip
                    else:
                        imgs = images
                    
                    # Apply scale
                    if scale != 1.0:
                        imgs = nn.functional.interpolate(
                            imgs, scale_factor=scale, mode='bilinear', align_corners=False
                        )
                    
                    # Forward pass
                    if view_type == 'query':
                        out, _ = model(imgs, None)
                    else:  # dictionary
                        _, out = model(None, imgs)
                    
                    if batch_features is None:
                        batch_features = out
                    else:
                        batch_features = batch_features + out
            
            # Average over augmentations
            batch_features = batch_features / (len(scales) * 2)
            
            # L2 normalize
            if len(batch_features.shape) == 3:
                # Part features: (B, D, P) -> flatten and normalize
                fnorm = torch.norm(batch_features, p=2, dim=1, keepdim=True) * np.sqrt(batch_features.size(-1))
                batch_features = batch_features / fnorm
                batch_features = batch_features.view(batch_size, -1)
            else:
                fnorm = torch.norm(batch_features, p=2, dim=1, keepdim=True)
                batch_features = batch_features / fnorm
            
            features.append(batch_features.cpu())
            labels.append(batch_labels)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels


# ============================================================================
# Voting-based Reranking
# ============================================================================

def voting_rerank(
    query_feature: torch.Tensor,
    query_label: int,
    dict_features: torch.Tensor,
    dict_labels: np.ndarray
) -> List[int]:
    """Voting-based reranking for a single query.
    
    This is the core algorithm from the paper. For each character label that 
    appears before the first correct match, we compute its average rank across
    all dictionary entries with that label. Labels are then re-ranked by this
    average rank score.
    
    Args:
        query_feature: Query feature vector (D,).
        query_label: Ground truth label ID.
        dict_features: Dictionary features (N, D).
        dict_labels: Dictionary label IDs (N,).
        
    Returns:
        new_rank: Binary list where position i is 1 if the query's true label
                  appears in top-(i+1) after voting rerank.
    """
    if not isinstance(dict_labels, np.ndarray):
        dict_labels = np.array(dict_labels)
    
    # Compute cosine similarity
    query = query_feature.view(-1, 1)
    scores = torch.mm(dict_features, query).squeeze(1).cpu().numpy()
    
    # Sort by similarity (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    # Find positions of correct matches
    correct_positions = np.argwhere(dict_labels == query_label).flatten()
    if len(correct_positions) == 0:
        return [0] * len(sorted_indices)
    
    # Find first correct match in sorted list
    sorted_labels = dict_labels[sorted_indices]
    correct_mask = np.in1d(sorted_indices, correct_positions)
    first_correct_pos = np.argwhere(correct_mask).flatten()
    
    if len(first_correct_pos) == 0:
        return [0] * len(sorted_indices)
    
    first_correct_pos = first_correct_pos[0]
    
    # Voting: compute average rank for each label before first correct match
    seen_labels = []
    label_scores = {}
    
    for i in range(first_correct_pos + 1):
        current_label = sorted_labels[i]
        
        if current_label not in seen_labels:
            seen_labels.append(current_label)
            
            # Find all positions of this label in sorted list
            label_positions = np.argwhere(sorted_labels == current_label).flatten()
            
            # Average rank score
            label_scores[current_label] = np.mean(label_positions)
    
    # Re-rank labels by average rank score
    ranked_labels = sorted(label_scores.keys(), key=lambda x: label_scores[x])
    
    # Find new rank of query label
    new_label_rank = len(ranked_labels)  # Default to worst case
    for i, label in enumerate(ranked_labels):
        if label == query_label:
            new_label_rank = i
            break
    
    # Generate binary rank list
    new_rank = [0] * len(sorted_indices)
    for i in range(len(sorted_indices)):
        if i >= new_label_rank:
            new_rank[i] = 1
    
    return new_rank


def compute_topn_accuracy(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    dict_features: torch.Tensor,
    dict_labels: torch.Tensor,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Compute Top-N accuracy with voting-based reranking.
    
    Args:
        query_features: Query features (Q, D).
        query_labels: Query labels (Q,).
        dict_features: Dictionary features (N, D).
        dict_labels: Dictionary labels (N,).
        device: Device for computation.
        
    Returns:
        Dictionary containing Top-N accuracy for N = 1, 5, 10, 50, 100.
    """
    query_features = query_features.to(device)
    dict_features = dict_features.to(device)
    
    query_labels_np = query_labels.numpy()
    dict_labels_np = dict_labels.numpy()
    
    num_queries = len(query_labels)
    num_gallery = len(dict_labels)
    
    # Aggregate voting results
    rank_results = np.zeros(num_gallery)
    
    print("Computing Top-N accuracy with voting rerank...")
    for i in tqdm(range(num_queries), desc="Voting rerank"):
        new_rank = voting_rerank(
            query_features[i],
            query_labels_np[i],
            dict_features,
            dict_labels_np
        )
        rank_results += np.array(new_rank)
    
    # Average over all queries
    rank_results = rank_results / num_queries
    
    # Extract Top-N accuracy
    results = {
        'Top-1': rank_results[0] * 100,
        'Top-5': rank_results[min(4, num_gallery - 1)] * 100,
        'Top-10': rank_results[min(9, num_gallery - 1)] * 100,
        'Top-50': rank_results[min(49, num_gallery - 1)] * 100,
        'Top-100': rank_results[min(99, num_gallery - 1)] * 100,
    }
    
    return results


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
    
    # Setup transforms
    h, w = config.get('height', 256), config.get('width', 256)
    pad = config.get('pad', 0)
    scales = config.get('scales', [1.0])
    
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
    
    # Extract features
    start_time = time.time()
    
    query_features, query_labels = extract_features(
        model, query_loader, 'query', device, scales
    )
    dict_features, dict_labels = extract_features(
        model, dict_loader, 'dictionary', device, scales
    )
    
    extraction_time = time.time() - start_time
    print(f"Feature extraction completed in {extraction_time:.1f}s")
    
    # Compute Top-N accuracy
    results = compute_topn_accuracy(
        query_features, query_labels,
        dict_features, dict_labels,
        device
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Retrieval Results (with Voting Rerank)")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}%")
    print("=" * 50)
    
    # Save results
    output_dir = config.get('output_dir', './results')
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
        for metric, value in results.items():
            f.write(f"{metric}: {value:.2f}%\n")
    
    print(f"\nResults saved to: {output_file}")
    
    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.1f}s")


if __name__ == '__main__':
    main()

