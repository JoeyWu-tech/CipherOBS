"""
Miscellaneous utilities for distributed training.
"""

import torch
import torch.distributed as dist
from torch import inf


class NativeScalerWithGradNormCount:
    """
    Native PyTorch gradient scaler with gradient norm counting.
    Used for mixed precision training.
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=3.0, parameters=None, 
                 create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the gradient norm of parameters.
    
    Args:
        parameters: Model parameters
        norm_type: Type of norm to compute
        
    Returns:
        Total gradient norm
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    
    if len(parameters) == 0:
        return torch.tensor(0.)
    
    device = parameters[0].grad.device
    
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
    
    return total_norm


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get the number of distributed processes."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_reduce_mean(x):
    """
    Reduce a value across all processes by averaging.
    
    Args:
        x: Value to reduce
        
    Returns:
        Average value across all processes
    """
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

