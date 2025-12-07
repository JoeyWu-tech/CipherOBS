"""FontDiffuser model for Stage 2 stroke refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .ids_encoder import IDSEncoder


class Upsample(nn.Module):
    """Upsampling module with optional convolution."""
    
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class FontDiffuserModel(ModelMixin, ConfigMixin):
    """FontDiffuser model with content encoder, style encoder and UNet.
    
    This model integrates IDS (Ideographic Description Sequence) encoding
    for Chinese character structure representation.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        ids_style_encoder=None,
        ids_path=None,
        glyph_path=None,
    ):
        """Initialize FontDiffuserModel.
        
        Args:
            unet: UNet model for diffusion.
            style_encoder: Style encoder model (unused, kept for compatibility).
            content_encoder: Content encoder model.
            ids_style_encoder: IDS-based style encoder.
            ids_path: Path to IDS dictionary file.
            glyph_path: Path to glyph definitions file.
        """
        super().__init__()
        self.unet = unet
        self.content_encoder = content_encoder
        
        # Initialize IDS encoder
        self.ids_encoder = IDSEncoder(ids_path, glyph_path, tensor_size=32)

        # Embedding layers for IDS tokens
        num_tokens = 445
        num_features = 64
        self.emb = nn.Embedding(num_tokens + 1, num_features)
        self.emb_flatten = nn.Sequential(
            Upsample(num_features, True, dims=2, out_channels=num_features // 2),
            Upsample(num_features // 2, True, dims=2, out_channels=num_features // 4),
        )
        self.num_features = num_features
        self.ids_style_encoder = ids_style_encoder

    def forward(
        self,
        x_t,
        timesteps,
        style_images,
        content_images,
        content_encoder_downsample_size,
        fantizi_list=None,
    ):
        """Forward pass of the model.
        
        Args:
            x_t: Noisy target images at timestep t.
            timesteps: Current diffusion timesteps.
            style_images: Style reference images.
            content_images: Content input images.
            content_encoder_downsample_size: Downsample size for content encoder.
            fantizi_list: List of Chinese characters for IDS encoding.
            
        Returns:
            Tuple of (noise_pred, offset_out_sum).
        """
        # Encode IDS representations
        batch = []
        for fantizi in fantizi_list:
            y, y_mask = self.ids_encoder.encode_char(fantizi)
            batch.append([y, y_mask])

        y_list = []
        y_mask_list = []
        max_len = max([len(y) for y, y_mask in batch])
        
        for y, y_mask in batch:
            if max_len > len(y):
                temp = np.zeros((max_len - len(y), y.shape[1], y.shape[2]), dtype=int)
                y_list.append(np.concatenate((y, temp), axis=0))
                y_mask_list.append(np.concatenate((y_mask, temp), axis=0))
            else:
                y_list.append(y)
                y_mask_list.append(y_mask)
        
        y = np.stack(y_list)
        y_mask = np.stack(y_mask_list)
        y = torch.from_numpy(y).to(x_t.device)
        y_mask = torch.from_numpy(y_mask).to(x_t.device)

        # Compute IDS embeddings
        y_emb = self.emb(y)
        y_mask = y_mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_features)
        y_emb = y_emb * y_mask
        y_emb = y_emb.sum(1)
        y_emb = y_emb.permute(0, 3, 1, 2).contiguous()
        y_emb = self.emb_flatten(y_emb)

        # Get style features from IDS encoder
        style_img_feature, _, _ = self.ids_style_encoder(y_emb)
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content features
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        
        # Get content features from style reference
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        # Prepare hidden states for UNet
        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features
        ]

        # Run UNet
        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        return noise_pred, offset_out_sum


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """FontDiffuser model for DPM-Solver sampling.
    
    This variant is optimized for inference with DPM-Solver scheduler.
    """

    @register_to_config
    def __init__(
        self,
        unet,
        style_encoder,
        content_encoder,
        ids_style_encoder,
        ids_path=None,
        glyph_path=None,
    ):
        """Initialize FontDiffuserModelDPM.
        
        Args:
            unet: UNet model for diffusion.
            style_encoder: Style encoder model (unused, kept for compatibility).
            content_encoder: Content encoder model.
            ids_style_encoder: IDS-based style encoder.
            ids_path: Path to IDS dictionary file.
            glyph_path: Path to glyph definitions file.
        """
        super().__init__()
        self.unet = unet
        self.content_encoder = content_encoder
        
        # Initialize IDS encoder
        self.ids_encoder = IDSEncoder(ids_path, glyph_path, tensor_size=32)

        # Embedding layers for IDS tokens
        num_tokens = 445
        num_features = 64
        self.emb = nn.Embedding(num_tokens + 1, num_features)
        self.emb_flatten = nn.Sequential(
            Upsample(num_features, True, dims=2, out_channels=num_features // 2),
            Upsample(num_features // 2, True, dims=2, out_channels=num_features // 4),
        )
        self.num_features = num_features
        self.ids_style_encoder = ids_style_encoder

    def forward(
        self,
        x_t,
        timesteps,
        cond,
        content_encoder_downsample_size,
        version,
    ):
        """Forward pass for DPM-Solver sampling.
        
        Args:
            x_t: Noisy images at timestep t.
            timesteps: Current diffusion timesteps.
            cond: Tuple of (content_images, style_images, fantizi_list).
            content_encoder_downsample_size: Downsample size for content encoder.
            version: Model version string.
            
        Returns:
            Predicted noise.
        """
        content_images = cond[0]
        style_images = cond[1]
        fantizi_list = cond[2]

        # Encode IDS representations
        batch = []
        for fantizi in fantizi_list:
            y, y_mask = self.ids_encoder.encode_char(fantizi)
            batch.append([y, y_mask])

        y_list = []
        y_mask_list = []
        max_len = max([len(y) for y, y_mask in batch])
        
        for y, y_mask in batch:
            if max_len > len(y):
                temp = np.zeros((max_len - len(y), y.shape[1], y.shape[2]), dtype=int)
                y_list.append(np.concatenate((y, temp), axis=0))
                y_mask_list.append(np.concatenate((y_mask, temp), axis=0))
            else:
                y_list.append(y)
                y_mask_list.append(y_mask)
        
        y = np.stack(y_list)
        y_mask = np.stack(y_mask_list)
        y = torch.from_numpy(y).to(x_t.device)
        y_mask = torch.from_numpy(y_mask).to(x_t.device)

        # Compute IDS embeddings
        y_emb = self.emb(y)
        y_mask = y_mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.num_features)
        y_emb = y_emb * y_mask
        y_emb = y_emb.sum(1)
        y_emb = y_emb.permute(0, 3, 1, 2).contiguous()
        y_emb = self.emb_flatten(y_emb)

        # Get style features
        style_img_feature, _, _ = self.ids_style_encoder(y_emb)
        batch_size, channel, height, width = style_img_feature.shape
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channel
        )

        # Get content features
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        
        # Get content features from style reference
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        style_content_res_features.append(style_content_feature)

        # Prepare hidden states
        input_hidden_states = [
            style_img_feature,
            content_residual_features,
            style_hidden_states,
            style_content_res_features
        ]

        # Run UNet
        out = self.unet(
            x_t,
            timesteps,
            encoder_hidden_states=input_hidden_states,
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]

        return noise_pred

