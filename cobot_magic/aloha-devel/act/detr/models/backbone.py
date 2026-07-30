# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
import os
import sys

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out



class DINOv2DenseBackbone(nn.Module):
    """
    Local DINOv2 dense backbone for real-robot ACT.
    It returns a dict compatible with Joiner: {"0": feature_map}.
    Input image is expected to be ImageNet-normalized by ACTPolicy.
    """
    def __init__(self, name='dinov2_vits14', repo=None, weights=None, train_layers=8, pool=2):
        super().__init__()
        self.name = name
        self.repo = repo or os.environ.get('DINOV2_REPO', '/home/d510/cobot_magic/dinov2_local/dinov2-main')
        self.weights = weights or os.environ.get('DINOV2_WEIGHTS', '/home/d510/cobot_magic/dinov2_local/dinov2_vits14_pretrain.pth')
        self.train_layers = int(train_layers)
        self.pool = max(1, int(pool))
        self.num_channels = 384 if 'vits14' in name else 768

        if not os.path.isdir(self.repo):
            raise FileNotFoundError(f'DINOv2 repo not found: {self.repo}')
        self.model = torch.hub.load(self.repo, name, source='local', pretrained=False)
        if os.path.isfile(self.weights):
            sd = torch.load(self.weights, map_location='cpu')
            if isinstance(sd, dict) and 'model' in sd:
                sd = sd['model']
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            msg = self.model.load_state_dict(sd, strict=False)
            print(f'[DINOv2] loaded weights: {self.weights}; status={msg}')
        else:
            print(f'[DINOv2][WARN] weights not found: {self.weights}; using repo defaults')

        # Freeze by default, then unfreeze last train_layers transformer blocks.
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.train_layers > 0 and hasattr(self.model, 'blocks'):
            for blk in self.model.blocks[-self.train_layers:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if hasattr(self.model, 'norm'):
                for p in self.model.norm.parameters():
                    p.requires_grad_(True)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f'[DINOv2] {name}, train_layers={self.train_layers}, pool={self.pool}, trainable={trainable/1e6:.2f}M / {total/1e6:.2f}M')

    def forward(self, x):
        # DINOv2 ViT-S/14 requires image size to be divisible by 14.
        # 476x630 = 34x45 patches and stays close to the 480x640 real camera stream.
        x = F.interpolate(x, size=(476, 630), mode='bilinear', align_corners=False)
        n = max(1, self.train_layers)
        try:
            outs = self.model.get_intermediate_layers(x, n=n, reshape=True)
            feat = torch.stack(outs, dim=0).mean(dim=0)
        except Exception:
            feats = self.model.forward_features(x)
            if isinstance(feats, dict) and 'x_norm_patchtokens' in feats:
                tokens = feats['x_norm_patchtokens']
            else:
                tokens = feats[:, 1:, :]
            B, N, C = tokens.shape
            h, w = 34, 45
            feat = tokens.transpose(1, 2).reshape(B, C, h, w)
        if self.pool > 1:
            feat = F.avg_pool2d(feat, kernel_size=self.pool, stride=self.pool)
        return {'0': feat}

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    name = args.backbone
    if str(name).startswith('dinov2'):
        backbone = DINOv2DenseBackbone(
            name=name,
            repo=getattr(args, 'dinov2_repo', None),
            weights=getattr(args, 'dinov2_weights', None),
            train_layers=getattr(args, 'dinov2_train_layers', 8),
            pool=getattr(args, 'dinov2_pool', 2),
        )
    else:
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
        #                             RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [4, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [4, 1]),
                                    RestNetBasicBlock(256, 256, 1))
        self.num_channels = 256
        # self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
        #                             RestNetBasicBlock(512, 512, 1))

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #
        # self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv1(x)
        # out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out
