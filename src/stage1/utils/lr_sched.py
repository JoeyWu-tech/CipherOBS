"""
Learning rate scheduling utilities.
"""

import math


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust learning rate with warmup and optional cosine decay.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch (can be fractional)
        args: Configuration with lr_schedule, warmup_epochs, lr, min_lr, epochs
    """
    if epoch < args.warmup_epochs:
        # Linear warmup
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            # Cosine annealing after warmup
            min_lr = getattr(args, 'min_lr', 0)
            lr = min_lr + (args.lr - min_lr) * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
            )
        else:
            raise NotImplementedError(f"Learning rate schedule '{args.lr_schedule}' not implemented")
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
    return lr

