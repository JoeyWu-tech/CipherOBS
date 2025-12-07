"""Helper functions for Stage 2 training and inference."""

import yaml
import torch
import torchvision.transforms as transforms


def save_args_to_yaml(args, output_file):
    """Save configuration arguments to a YAML file.
    
    Args:
        args: Namespace containing configuration arguments.
        output_file: Path to output YAML file.
    """
    args_dict = vars(args)
    
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Compute x_0 from predicted noise.
    
    Args:
        scheduler: Noise scheduler instance.
        noise_pred: Predicted noise tensor.
        x_t: Noisy sample at timestep t.
        timesteps: Current timesteps.
        
    Returns:
        Predicted original sample x_0.
    """
    batch_size = noise_pred.shape[0]
    
    for i in range(batch_size):
        noise_pred_i = noise_pred[i][None, :]
        t = timesteps[i]
        x_t_i = x_t[i][None, :]
        
        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            pred_original_sample = torch.cat(
                (pred_original_sample, pred_original_sample_i), dim=0
            )
    
    return pred_original_sample


def reNormalize_img(pred_original_sample):
    """Re-normalize image from [-1, 1] to [0, 1].
    
    Args:
        pred_original_sample: Image tensor normalized to [-1, 1].
        
    Returns:
        Image tensor normalized to [0, 1].
    """
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)
    return pred_original_sample


def normalize_mean_std(image):
    """Apply ImageNet normalization to image.
    
    Args:
        image: Input image tensor.
        
    Returns:
        Normalized image tensor.
    """
    transforms_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image = transforms_norm(image)
    return image

