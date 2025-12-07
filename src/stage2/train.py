"""
Stage 2: Stroke Refinement (SR) Training Script
Decoding Ancient Oracle Bone Script via Generative Dictionary Retrieval
"""

import os
import math
import time
import logging
import argparse
from tqdm.auto import tqdm

import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from data import FontDataset, CollateFN
from models import (
    FontDiffuserModel,
    StyleEncoder,
    build_unet,
    build_style_encoder,
    build_content_encoder,
    build_ddpm_scheduler,
)
from models.criterion import ContentPerceptualLoss
from utils import save_args_to_yaml, x0_from_epsilon, reNormalize_img, normalize_mean_std

os.environ['NCCL_P2P_DISABLE'] = '1'

logger = get_logger(__name__)


def dict2namespace(config):
    """Convert dictionary to namespace for attribute-style access."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_args_from_config(config, project_root):
    """Build args namespace from config dictionary."""
    args = argparse.Namespace()
    
    # Experiment settings
    args.seed = config['experiment']['seed']
    args.experience_name = config['experiment']['name']
    args.output_dir = os.path.join(project_root, config['experiment']['output_dir'])
    args.logging_dir = config['experiment']['logging_dir']
    args.report_to = config['experiment']['report_to']
    
    # Data settings
    args.data_root = os.path.join(project_root, config['data']['root'])
    args.train_phase = config['data']['train_phase']
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
    
    # Training settings
    args.train_batch_size = config['training']['batch_size']
    args.max_train_steps = config['training']['max_train_steps']
    args.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    args.ckpt_interval = config['training']['ckpt_interval']
    args.log_interval = config['training']['log_interval']
    args.phase_1_ckpt_dir = config['training'].get('phase_1_ckpt_dir')
    if args.phase_1_ckpt_dir:
        args.phase_1_ckpt_dir = os.path.join(project_root, args.phase_1_ckpt_dir)
    
    # Phase 2 SCR settings
    args.phase_2 = config['training']['phase_2']
    args.scr_ckpt_path = config['training'].get('scr_ckpt_path')
    if args.scr_ckpt_path:
        args.scr_ckpt_path = os.path.join(project_root, args.scr_ckpt_path)
    args.temperature = config['training']['temperature']
    args.mode = config['training']['mode']
    args.scr_image_size = config['training']['scr_image_size']
    args.num_neg = config['training']['num_neg']
    args.nce_layers = config['training']['nce_layers']
    args.sc_coefficient = config['training']['sc_coefficient']
    
    # Loss coefficients
    args.perceptual_coefficient = config['loss']['perceptual_coefficient']
    args.offset_coefficient = config['loss']['offset_coefficient']
    
    # Optimizer settings
    args.learning_rate = config['optimizer']['learning_rate']
    args.scale_lr = config['optimizer']['scale_lr']
    args.adam_beta1 = config['optimizer']['adam_beta1']
    args.adam_beta2 = config['optimizer']['adam_beta2']
    args.adam_weight_decay = config['optimizer']['adam_weight_decay']
    args.adam_epsilon = config['optimizer']['adam_epsilon']
    args.max_grad_norm = config['optimizer']['max_grad_norm']
    
    # LR scheduler settings
    args.lr_scheduler = config['lr_scheduler']['type']
    args.lr_warmup_steps = config['lr_scheduler']['warmup_steps']
    
    # Classifier-free guidance
    args.drop_prob = config['classifier_free']['drop_prob']
    args.beta_scheduler = config['classifier_free']['beta_scheduler']
    
    # Mixed precision
    args.mixed_precision = config['mixed_precision']
    
    return args


def main():
    parser = argparse.ArgumentParser(description="Stage 2 SR Training")
    parser.add_argument(
        "--config",
        default='configs/stage2/train.yaml',
        type=str,
        help="Path to the config file (relative to project root)"
    )
    parser.add_argument('--local_rank', default=-1, type=int)
    cmd_args = parser.parse_args()
    
    # Load configuration
    project_root = get_project_root()
    config_path = os.path.join(project_root, cmd_args.config)
    config = load_config(config_path)
    args = build_args_from_config(config, project_root)
    
    # Setup logging directory
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "stage2_training.log"),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Build model components
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    
    # Load Phase 1 checkpoint if provided
    if args.phase_1_ckpt_dir:
        unet.load_state_dict(torch.load(os.path.join(args.phase_1_ckpt_dir, "unet.pth")))
        style_encoder.load_state_dict(torch.load(os.path.join(args.phase_1_ckpt_dir, "style_encoder.pth")))
        content_encoder.load_state_dict(torch.load(os.path.join(args.phase_1_ckpt_dir, "content_encoder.pth")))
    
    # Build IDS style encoder
    ids_style_encoder = StyleEncoder(
        G_ch=64,
        resolution=128,
        input_nc=16
    )
    
    # Build FontDiffuser model
    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        ids_style_encoder=ids_style_encoder,
        ids_path=args.ids_path,
        glyph_path=args.glyph_path
    )
    
    # Build perceptual loss
    perceptual_loss = ContentPerceptualLoss()
    
    # Load SCR module for Phase 2 training
    if args.phase_2:
        from models.modules import SCR
        from models.build import build_scr
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)
    
    # Setup data transforms
    content_transforms = transforms.Compose([
        transforms.Resize(
            args.content_image_size,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    style_transforms = transforms.Compose([
        transforms.Resize(
            args.style_image_size,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    target_transforms = transforms.Compose([
        transforms.Resize(
            (args.resolution, args.resolution),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Build dataset and dataloader
    train_dataset = FontDataset(
        args=args,
        phase='train',
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=args.phase_2
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN()
    )
    
    # Scale learning rate if needed
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes
        )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if args.phase_2:
        scr = scr.to(accelerator.device)
    
    # Initialize trackers
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(
            args=args,
            output_file=os.path.join(args.output_dir, f"{args.experience_name}_config.yaml")
        )
    
    # Setup progress bar
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Training")
    
    # Calculate epochs
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Training loop
    global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        
        for step, samples in enumerate(train_dataloader):
            model.train()
            
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]
            fantizi_list = samples['fantizi']
            
            with accelerator.accumulate(model):
                # Sample noise
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps, (bsz,),
                    device=target_images.device
                )
                timesteps = timesteps.long()
                
                # Add noise to target images (forward diffusion)
                noisy_target_images = noise_scheduler.add_noise(
                    target_images, noise, timesteps
                )
                
                # Classifier-free training: randomly drop conditioning
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value == 1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1
                
                # Forward pass
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    fantizi_list=fantizi_list
                )
                
                # Compute losses
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                
                # Compute perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps
                )
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device
                )
                
                # Total loss
                loss = (
                    diff_loss +
                    args.perceptual_coefficient * percep_loss +
                    args.offset_coefficient * offset_loss
                )
                
                # Add SCR loss for Phase 2
                if args.phase_2:
                    neg_images = samples["neg_images"]
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm,
                        target_images,
                        neg_images,
                        nce_layers=args.nce_layers
                    )
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings
                    )
                    loss += args.sc_coefficient * sc_loss
                
                # Gather losses for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                # Save checkpoint
                if accelerator.is_main_process and global_step % args.ckpt_interval == 0:
                    save_dir = os.path.join(args.output_dir, f"global_step_{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    torch.save(
                        model.module.unet.state_dict(),
                        os.path.join(save_dir, "unet.pth")
                    )
                    torch.save(
                        model.module.content_encoder.state_dict(),
                        os.path.join(save_dir, "content_encoder.pth")
                    )
                    torch.save(
                        model.module.ids_style_encoder.state_dict(),
                        os.path.join(save_dir, "ids_style_encoder.pth")
                    )
                    torch.save(model.module, os.path.join(save_dir, "total_model.pth"))
                    
                    logging.info(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                        f"Checkpoint saved at step {global_step}"
                    )
            
            # Log progress
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                logging.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                    f"Step {global_step} => loss = {loss}"
                )
            progress_bar.set_postfix(**logs)
            
            if global_step >= args.max_train_steps:
                break
    
    accelerator.end_training()
    
    if accelerator.is_main_process:
        logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()

