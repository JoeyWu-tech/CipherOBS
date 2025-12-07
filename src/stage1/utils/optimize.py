"""
Optimizer utilities.
"""

import torch.optim as optim


def get_optimizer(config, parameters):
    """
    Create optimizer based on configuration.
    
    Args:
        config: Configuration object with optimizer settings
        parameters: Model parameters to optimize
        
    Returns:
        Configured optimizer instance
    """
    if config.optim.optimizer == 'Adam':
        return optim.Adam(
            parameters, 
            lr=config.optim.lr, 
            weight_decay=config.optim.weight_decay,
            betas=(0.9, 0.999), 
            amsgrad=config.optim.amsgrad, 
            eps=config.optim.eps
        )
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(
            parameters, 
            lr=config.optim.lr, 
            weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(
            parameters, 
            lr=config.optim.lr, 
            momentum=0.9
        )
    else:
        raise NotImplementedError(f'Optimizer {config.optim.optimizer} not supported')

