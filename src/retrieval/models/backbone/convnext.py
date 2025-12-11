# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.
#
# ConvNeXt: A ConvNet for the 2020s
# Paper: https://arxiv.org/abs/2201.03545

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    """ConvNeXt Block.
    
    Implementation using DwConv -> Permute -> LayerNorm -> Linear -> GELU -> Linear -> Permute.
    
    Args:
        dim: Number of input channels.
        drop_path: Stochastic depth rate. Default: 0.0
        layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = shortcut + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm supporting both channels_last and channels_first data formats.
    
    Args:
        normalized_shape: Input shape from an expected input.
        eps: A value added to the denominator for numerical stability.
        data_format: 'channels_last' (N, H, W, C) or 'channels_first' (N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    """ConvNeXt backbone for feature extraction.
    
    Paper: A ConvNet for the 2020s (https://arxiv.org/abs/2201.03545)
    
    Args:
        in_chans: Number of input image channels. Default: 3
        num_classes: Number of classes for classification head. Default: 1000
        depths: Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims: Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate: Stochastic depth rate. Default: 0.
        layer_scale_init_value: Init value for Layer Scale. Default: 1e-6.
        head_init_scale: Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        head_init_scale=1.,
    ):
        super().__init__()

        # Stem and downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Feature resolution stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """Extract features from input image.
        
        Returns:
            gap_feature: Global average pooled feature (B, C)
            spatial_feature: Spatial feature map (B, C, H, W)
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        gap_feature = self.norm(x.mean([-2, -1]))
        return gap_feature, x

    def forward(self, x):
        return self.forward_features(x)


# Pre-trained model URLs from Facebook Research
MODEL_URLS = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth",
}


def convnext_tiny(pretrained=True, in_22k=True, **kwargs):
    """ConvNeXt-Tiny model with 768-dim output features."""
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_tiny_22k'] if in_22k else MODEL_URLS['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def convnext_small(pretrained=True, in_22k=True, **kwargs):
    """ConvNeXt-Small model with 768-dim output features."""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_small_22k'] if in_22k else MODEL_URLS['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


def convnext_base(pretrained=True, in_22k=True, **kwargs):
    """ConvNeXt-Base model with 1024-dim output features."""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = MODEL_URLS['convnext_base_22k'] if in_22k else MODEL_URLS['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    return model



