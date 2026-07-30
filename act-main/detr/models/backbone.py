# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.

DINOv2-ACT Backbone V5 (Innovation-1 Stable Finetune):
1. Load local DINOv2 ViT-S/14 from dinov2-main and dinov2_vits14_pretrain.pth.
2. Automatically handle ACT/ImageNet-normalized, 0~1 RGB, and 0~255 RGB inputs.
3. Pad images to multiples of patch_size=14 without resizing or geometry distortion.
4. Support full-resolution patch grid or optional 2x2 avg-pooling by --dinov2_pool.
5. Support partial DINOv2 fine-tuning by --dinov2_train_layers:
   - 0  : freeze all DINOv2 parameters
   - N>0: train last N transformer blocks + norm
   - -1 : train all DINOv2 parameters
"""

import os
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Kept for compatibility with the original ACT/DETR code.
    """
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)

        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class DINOv2Backbone(nn.Module):
    """DINOv2 ViT-S/14 backbone for ACT."""
    def __init__(self, train_backbone: bool, train_layers: int = 4, pool_stride: int = 1):
        super().__init__()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        act_root = os.path.dirname(os.path.dirname(current_dir))

        dinov2_local_path = os.path.join(act_root, 'dinov2-main')
        weight_path = os.path.join(act_root, 'dinov2_vits14_pretrain.pth')

        print(f'\n🚀 [Ours V5] Loading local DINOv2 source: {dinov2_local_path}')
        self.body = torch.hub.load(
            dinov2_local_path,
            'dinov2_vits14',
            source='local',
            pretrained=False,
        )

        print(f'📦 [Ours V5] Loading local DINOv2 weights: {weight_path}')
        if os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, weights_only=True, map_location='cpu')
            except TypeError:
                state_dict = torch.load(weight_path, map_location='cpu')
            self.body.load_state_dict(state_dict)
            print('✅ [Ours V5] DINOv2 local weights loaded successfully!')
        else:
            raise FileNotFoundError(f'[Ours V5 ERROR] Cannot find DINOv2 weights: {weight_path}')

        self.num_channels = 384
        self.patch_size = 14
        self.pool_stride = int(pool_stride)
        if self.pool_stride < 1:
            raise ValueError(f'dinov2_pool must be >= 1, got {self.pool_stride}')

        self.register_buffer(
            'pixel_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'pixel_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self._configure_trainable_layers(train_backbone=train_backbone, train_layers=train_layers)
        self._print_trainable_summary(train_layers=train_layers)

    def _configure_trainable_layers(self, train_backbone: bool, train_layers: int):
        # Start from fully frozen; selectively unfreeze only what we need.
        for parameter in self.body.parameters():
            parameter.requires_grad_(False)

        if not train_backbone:
            return

        train_layers = int(train_layers)
        if train_layers == 0:
            return

        if train_layers < 0:
            for parameter in self.body.parameters():
                parameter.requires_grad_(True)
            return

        if not hasattr(self.body, 'blocks'):
            raise AttributeError('DINOv2 model does not expose .blocks; cannot apply partial finetuning.')

        num_blocks = len(self.body.blocks)
        train_layers = min(train_layers, num_blocks)

        for block in self.body.blocks[-train_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

        if hasattr(self.body, 'norm'):
            for parameter in self.body.norm.parameters():
                parameter.requires_grad_(True)

    def _print_trainable_summary(self, train_layers: int):
        total_params = sum(p.numel() for p in self.body.parameters())
        trainable_params = sum(p.numel() for p in self.body.parameters() if p.requires_grad)
        print(
            f'🔧 [Ours V5] DINOv2 train_layers={train_layers}, pool_stride={self.pool_stride}, '
            f'trainable={trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M '
            f'({100.0 * trainable_params / max(total_params, 1):.2f}%)\n'
        )

    def _detect_input_mode_once(self, x: torch.Tensor):
        # This GPU sync happens only once; subsequent batches reuse the same mode.
        x_min = x.min().item()
        x_max = x.max().item()

        print('🔍 [DINOv2 Debug] input range before DINO preprocess:', x_min, x_max, x.shape)
        if x_min < 0.0:
            self._input_mode = 'imagenet_normalized'
            print('✅ [DINOv2 Debug] ACT input is already ImageNet-normalized. No extra normalization.')
        elif x_max > 10.0:
            self._input_mode = 'rgb_255'
            print('✅ [DINOv2 Debug] Detected 0~255 RGB. Applying /255 and ImageNet normalization.')
        else:
            self._input_mode = 'rgb_01'
            print('✅ [DINOv2 Debug] Detected 0~1 RGB. Applying ImageNet normalization.')

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, '_input_mode'):
            self._detect_input_mode_once(x)

        if self._input_mode == 'imagenet_normalized':
            return x

        if self._input_mode == 'rgb_255':
            x = x / 255.0

        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, tensor):
        if hasattr(tensor, 'tensors'):
            x = tensor.tensors
        else:
            x = tensor

        x = x.float()

        B, C, H, W = x.shape
        if C != 3:
            raise ValueError(f'DINOv2Backbone expects 3-channel RGB input, got C={C}')

        x = self._preprocess_input(x)

        new_H = math.ceil(H / self.patch_size) * self.patch_size
        new_W = math.ceil(W / self.patch_size) * self.patch_size
        pad_h = new_H - H
        pad_w = new_W - W

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        features = self.body.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']

        h_patches = new_H // self.patch_size
        w_patches = new_W // self.patch_size

        out_tensor = patch_tokens.transpose(1, 2).reshape(
            B,
            self.num_channels,
            h_patches,
            w_patches,
        )

        if self.pool_stride > 1:
            out_tensor = F.avg_pool2d(out_tensor, kernel_size=self.pool_stride, stride=self.pool_stride)

        return {'0': out_tensor}


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []

        for _, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = getattr(args, 'lr_backbone', 0.0) > 0
    train_layers = getattr(args, 'dinov2_train_layers', 4)
    pool_stride = getattr(args, 'dinov2_pool', 1)

    backbone = DINOv2Backbone(
        train_backbone=train_backbone,
        train_layers=train_layers,
        pool_stride=pool_stride,
    )

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
