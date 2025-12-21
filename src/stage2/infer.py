"""
Stage 2: Stroke Refinement (SR) Inference Script
Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval
"""

import os
import argparse

import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models import (
    FontDiffuserModelDPM,
    FontDiffuserDPMPipeline,
    StyleEncoder,
    build_unet,
    build_content_encoder,
    build_ddpm_scheduler,
)


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_latest_checkpoint(base_dir):
    """Find the checkpoint with the largest global step number.
    
    Args:
        base_dir: Base directory containing checkpoint folders (e.g., global_step_1000/).
        
    Returns:
        Path to the latest checkpoint directory.
        
    Raises:
        FileNotFoundError: If no valid checkpoint is found.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Checkpoint base directory not found: {base_dir}")
    
    # Find all global_step_* directories
    checkpoint_dirs = []
    for name in os.listdir(base_dir):
        if name.startswith("global_step_"):
            try:
                step = int(name.split("_")[-1])
                checkpoint_dirs.append((step, name))
            except ValueError:
                continue
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No valid checkpoints found in: {base_dir}")
    
    # Sort by step number and get the latest
    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_step, latest_name = checkpoint_dirs[0]
    latest_path = os.path.join(base_dir, latest_name)
    
    print(f"=> Auto-selected checkpoint: {latest_name} (step {latest_step})")
    
    return latest_path


def build_args_from_config(config, project_root):
    """Build args namespace from config dictionary."""
    args = argparse.Namespace()
    
    # Experiment settings
    args.seed = config['experiment']['seed']
    args.output_dir = os.path.join(project_root, config['experiment']['output_dir'])
    
    # Data settings
    args.input_dir = os.path.join(project_root, config['data']['input_dir'])
    args.style_image_path = os.path.join(project_root, config['data']['style_image_path'])
    args.ids_path = os.path.join(project_root, config['data']['ids_path'])
    args.glyph_path = os.path.join(project_root, config['data']['glyph_path'])
    
    # Model settings
    args.resolution = config['model']['resolution']
    args.unet_channels = tuple(config['model']['unet_channels'])
    args.style_image_size = (config['model']['style_image_size'], config['model']['style_image_size'])
    args.content_image_size = (config['model']['content_image_size'], config['model']['content_image_size'])
    args.content_encoder_downsample_size = config['model']['content_encoder_downsample_size']
    args.channel_attn = config['model']['channel_attn']
    args.content_start_channel = config['model']['content_start_channel']
    args.style_start_channel = config['model']['style_start_channel']
    
    # Handle checkpoint path: support both explicit path and auto-selection
    ckpt_dir = config['model']['ckpt_dir']
    if ckpt_dir == "auto":
        # Auto-select the latest checkpoint
        ckpt_base_dir = os.path.join(project_root, config['model']['ckpt_base_dir'])
        args.ckpt_dir = get_latest_checkpoint(ckpt_base_dir)
    else:
        # Use the explicitly specified checkpoint path
        args.ckpt_dir = os.path.join(project_root, ckpt_dir)
    
    # Sampling settings
    args.algorithm_type = config['sampling']['algorithm_type']
    args.guidance_type = config['sampling']['guidance_type']
    args.guidance_scale = config['sampling']['guidance_scale']
    args.num_inference_steps = config['sampling']['num_inference_steps']
    args.model_type = config['sampling']['model_type']
    args.order = config['sampling']['order']
    args.skip_type = config['sampling']['skip_type']
    args.method = config['sampling']['method']
    args.correcting_x0_fn = config['sampling'].get('correcting_x0_fn')
    args.t_start = config['sampling'].get('t_start')
    args.t_end = config['sampling'].get('t_end')
    args.beta_scheduler = "scaled_linear"
    
    # Output settings
    args.output_image_size = config['output']['image_size']
    
    return args


def load_fontdiffuser_pipeline(args, device):
    """Load the FontDiffuser pipeline for inference.
    
    Args:
        args: Configuration arguments.
        device: Device to load model on.
        
    Returns:
        FontDiffuserDPMPipeline instance.
    """
    # Build model components
    unet = build_unet(args=args)
    content_encoder = build_content_encoder(args=args)
    
    # Build IDS style encoder
    ids_style_encoder = StyleEncoder(
        G_ch=64,
        resolution=128,
        input_nc=16
    )
    
    # Build DPM model
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=None,
        content_encoder=content_encoder,
        ids_style_encoder=ids_style_encoder,
        ids_path=args.ids_path,
        glyph_path=args.glyph_path
    )
    
    # Load checkpoint
    # Note: weights_only=False is required because the checkpoint contains the full model object
    total_model = torch.load(os.path.join(args.ckpt_dir, "total_model.pth"), map_location='cpu', weights_only=False)
    model.unet.load_state_dict(total_model.unet.state_dict())
    model.content_encoder.load_state_dict(total_model.content_encoder.state_dict())
    model.ids_style_encoder.load_state_dict(total_model.ids_style_encoder.state_dict())
    model.emb.load_state_dict(total_model.emb.state_dict())
    model.emb_flatten.load_state_dict(total_model.emb_flatten.state_dict())
    
    model = model.to(device)
    model.eval()
    
    # Build noise scheduler
    noise_scheduler = build_ddpm_scheduler(args)
    
    # Build pipeline
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=noise_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale
    )
    
    return pipe


def run_inference(
    pipe,
    source_image,
    reference_image,
    hanzi,
    args,
    device
):
    """Run inference on a single image.
    
    Args:
        pipe: FontDiffuserDPMPipeline instance.
        source_image: Input content image (Stage 1 output).
        reference_image: Style reference image.
        hanzi: Chinese character for IDS encoding.
        args: Configuration arguments.
        device: Device to run inference on.
        
    Returns:
        Tuple of (output_image, intermediate_images).
    """
    # Prepare content transform
    content_transforms = transforms.Compose([
        transforms.Resize(
            args.content_image_size,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Prepare style transform
    style_transforms = transforms.Compose([
        transforms.Resize(
            args.style_image_size,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Process images
    content_image = content_transforms(source_image).unsqueeze(0).to(device)
    style_image = style_transforms(reference_image).unsqueeze(0).to(device)
    
    # Generate
    out_images, inter_images = pipe.generate(
        content_images=content_image,
        style_images=style_image,
        hanzi=[hanzi],
        batch_size=1,
        order=args.order,
        num_inference_step=args.num_inference_steps,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        algorithm_type=args.algorithm_type,
        skip_type=args.skip_type,
        method=args.method,
    )
    
    # Resize output
    out_image = out_images[0]
    desired_size = (args.output_image_size, args.output_image_size)
    out_image = out_image.resize(desired_size)
    
    # Resize intermediate images
    inter_images = [image[0].resize(desired_size) for image in inter_images]
    
    return out_image, inter_images


def extract_character_from_filename(filename):
    """Extract Chinese character from filename.
    
    Args:
        filename: Image filename.
        
    Returns:
        Extracted Chinese character or None.
    """
    import re
    chars = re.findall(r'[\u4e00-\u9fff]', filename)
    return chars[-1] if chars else None


def main():
    parser = argparse.ArgumentParser(description="Stage 2 SR Inference")
    parser.add_argument(
        "--config",
        default='configs/stage2/infer.yaml',
        type=str,
        help="Path to the config file (relative to project root)"
    )
    cmd_args = parser.parse_args()
    
    # Load configuration
    project_root = get_project_root()
    config_path = os.path.join(project_root, cmd_args.config)
    config = load_config(config_path)
    args = build_args_from_config(config, project_root)
    
    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")
    
    # Load pipeline
    print("=> Loading FontDiffuser pipeline...")
    pipe = load_fontdiffuser_pipeline(args, device)
    
    # Load style reference image
    reference_image = Image.open(args.style_image_path).convert("RGB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=> Saving results to: {args.output_dir}")
    
    # Process all images in input directory
    input_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"=> Processing {len(input_files)} images...")
    
    for filename in input_files:
        # Load source image
        source_path = os.path.join(args.input_dir, filename)
        source_image = Image.open(source_path).convert("RGB")
        
        # Extract character from filename
        hanzi = extract_character_from_filename(filename)
        if hanzi is None:
            print(f"Warning: Could not extract character from {filename}, skipping...")
            continue
        
        # Run inference
        out_image, _ = run_inference(
            pipe=pipe,
            source_image=source_image,
            reference_image=reference_image,
            hanzi=hanzi,
            args=args,
            device=device
        )
        
        # Save output
        output_path = os.path.join(args.output_dir, filename)
        out_image.save(output_path)
        print(f"=> Processed: {filename}")
    
    print("=> Inference completed successfully!")


if __name__ == "__main__":
    main()

