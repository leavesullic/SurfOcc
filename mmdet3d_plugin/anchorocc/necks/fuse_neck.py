import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmdet.models import NECKS


@NECKS.register_module()
class VoxelBEVFuser(BaseModule):
    def __init__(self,
                voxel_channels,
                bev_channels,
                init_cfg=None,
                coeff_bias=True):
        super().__init__(init_cfg=init_cfg)
        
        self.voxel_channels = voxel_channels
        self.bev_channels = bev_channels
        
        self.bev_norm_cfg = dict(type='BN2d', requires_grad=True)
        self.voxel_norm_cfg = dict(type='BN3d', requires_grad=True)

        self.voxel_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=self.voxel_channels,
                out_channels=self.voxel_channels,
                kernel_size=3,
                padding=1
            ),
            build_norm_layer(self.voxel_norm_cfg, self.voxel_channels)[1],
            nn.ReLU(inplace=True)
        )
        self.bev_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.bev_channels,
                out_channels=self.voxel_channels,
                kernel_size=3,
                padding=1),
            build_norm_layer(self.bev_norm_cfg, self.voxel_channels)[1],
            nn.ReLU(inplace=True)
        )
        self.fuse_coeff = nn.Conv3d(self.voxel_channels, 1, kernel_size=1, bias=coeff_bias)
    

    def forward(self, voxel_feat, bev_feat):

        voxel_feat = self.voxel_conv(voxel_feat)
        coeff = self.fuse_coeff(voxel_feat).sigmoid()

        bev_feat = self.bev_conv(bev_feat)
        
        fuse_voxel_feat = voxel_feat + coeff * bev_feat.unsqueeze(-1)

        return fuse_voxel_feat



