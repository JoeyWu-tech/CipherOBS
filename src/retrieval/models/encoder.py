# Copyright (c) CipherOBS Authors. All rights reserved.
"""
Feature Encoder for OBS Dictionary Retrieval.

This module implements the ConvNeXt-based feature encoder used for extracting
visual features from Oracle Bone Script images. The encoder incorporates
triplet attention for enhanced part-based feature learning.

Note: The model structure matches the original training code to ensure
weight compatibility with pre-trained checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model


# ============================================================================
# Attention Modules
# ============================================================================

class ZPool(nn.Module):
    """Z-Pool: Concatenates max and mean pooled features along channel dimension."""
    
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
            dim=1
        )


class BasicConv(nn.Module):
    """Basic convolution block with optional batch norm and ReLU."""
    
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class AttentionGate(nn.Module):
    """Spatial attention gate using Z-Pool and convolution."""
    
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
    
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    """Triplet Attention module for capturing cross-dimension interactions."""
    
    def __init__(self):
        super().__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
    
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        return x_out11, x_out21


# ============================================================================
# Classification Block
# ============================================================================

def weights_init_kaiming(m):
    """Initialize weights using Kaiming initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """Initialize classifier weights."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    """Classification block with bottleneck layer.
    
    Args:
        input_dim: Input feature dimension.
        class_num: Number of output classes.
        droprate: Dropout rate.
        relu: Whether to use ReLU.
        bnorm: Whether to use batch normalization.
        num_bottleneck: Bottleneck dimension.
        linear: Whether to use linear layer.
        return_f: Whether to return features during training.
    """
    
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, 
                 num_bottleneck=512, linear=True, return_f=False):
        super().__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    
    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


# ============================================================================
# Main Model (matches original structure for weight compatibility)
# ============================================================================

class build_convnext(nn.Module):
    """ConvNeXt-based feature encoder.
    
    Note: This class name and structure matches the original training code
    to ensure weight compatibility.
    
    Args:
        num_classes: Number of character classes.
        block: Number of part attention blocks. Default: 2.
        return_f: Return features during training. Default: False.
        resnet: Use ResNet backbone instead. Default: False.
    """
    
    def __init__(self, num_classes, block=2, return_f=False, resnet=False):
        super().__init__()
        self.return_f = return_f
        self.block = block
        
        if resnet:
            raise NotImplementedError("ResNet backbone not supported in this version")
        
        # ConvNeXt backbone
        convnext_name = "convnext_tiny"
        self.in_planes = 768
        self.convnext = create_model(convnext_name, pretrained=True)
        
        # Layer norm for global features
        self.norm = nn.LayerNorm(self.in_planes, eps=1e-6)
        
        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.tri_layer = TripletAttention()
        
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        # Extract features from backbone
        spatial_feature = self.convnext.forward_features(x)  # (B, C, H, W)
        gap_feature = self.norm(spatial_feature.mean([-2, -1]))  # (B, C)
        
        tri_features = self.tri_layer(spatial_feature)
        convnext_feature = self.classifier1(gap_feature)

        tri_list = []
        for i in range(self.block):
            tri_list.append(tri_features[i].mean([-2, -1]))
        triatten_features = torch.stack(tri_list, dim=2)
        
        if self.block == 0:
            y = []
        else:
            y = self.part_classifier(self.block, triatten_features, cls_name='classifier_mcb')

        if self.training:
            y = y + [convnext_feature]
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
        else:
            ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            y = torch.cat([y, ffeature], dim=2)

        return y

    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        """Forward through part classifiers."""
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


class two_view_net(nn.Module):
    """Two-view network for OBS retrieval.
    
    Uses shared weights between query and dictionary branches.
    
    Note: This class name matches the original training code for weight compatibility.
    
    Args:
        class_num: Number of character classes.
        block: Number of attention blocks. Default: 2.
        return_f: Return features during training. Default: False.
        resnet: Use ResNet backbone. Default: False.
    """
    
    def __init__(self, class_num, block=2, return_f=False, resnet=False):
        super().__init__()
        self.model_1 = build_convnext(
            num_classes=class_num, 
            block=block, 
            return_f=return_f, 
            resnet=resnet
        )

    def forward(self, x1, x2):
        """Forward pass for query and dictionary images.
        
        Args:
            x1: Query images (can be None).
            x2: Dictionary images (can be None).
            
        Returns:
            Tuple of (query_features, dict_features).
        """
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            y2 = self.model_1(x2)
        
        return y1, y2


# Alias for compatibility
FeatureEncoder = build_convnext
TwoViewEncoder = two_view_net
