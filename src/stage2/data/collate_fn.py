"""Custom collate function for Stage 2 dataloader."""

import torch


class CollateFN:
    """Custom collate function for batching font dataset samples."""
    
    def __init__(self):
        pass
    
    def __call__(self, batch):
        """Collate a batch of samples.
        
        Args:
            batch: List of sample dictionaries.
            
        Returns:
            Batched dictionary with stacked tensors.
        """
        batched_data = {}
        
        for k in batch[0].keys():
            batch_key_data = [ele[k] for ele in batch]
            if isinstance(batch_key_data[0], torch.Tensor):
                batch_key_data = torch.stack(batch_key_data)
            batched_data[k] = batch_key_data
        
        return batched_data

