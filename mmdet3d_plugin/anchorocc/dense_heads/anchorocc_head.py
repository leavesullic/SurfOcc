# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from projects.mmdet3d_plugin.anchorocc.utils.header import Header
from projects.mmdet3d_plugin.models.utils.bricks import run_time

@HEADS.register_module()
class AnchorOccHead(nn.Module):
    def __init__(
        self,
        bev_h,
        bev_w,
        bev_z,
        scene_size=None,
        pc_range=None,
        embed_dims=128,
        transformer=None,
        **kwargs
    ):
        super().__init__()
        self.scene_size = scene_size
        self.pc_range = pc_range
        
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_h = scene_size[0]
        self.real_w = scene_size[1]
        self.embed_dims = embed_dims
        self.voxel_embed = nn.Embedding((self.bev_h) * (self.bev_w) * (self.bev_z), self.embed_dims)

        # self.mask_embed = nn.Embedding(1, self.embed_dims)
        # self.positional_encoding = build_positional_encoding(positional_encoding)

        self.transformer = build_transformer(transformer)
        self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 
                                                        0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))


        
    def forward(self, mlvl_feats, voxel_feat, voxel_anchor, cam_params, img_metas):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth.
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        voxel_queries = self.voxel_embed.weight.to(dtype)
        voxel_queries = voxel_queries.unsqueeze(1).repeat(1, bs, 1)  # xyz bs dim

        # bev_pos_self_attn = self.positional_encoding(
        #     torch.zeros((bs, self.bev_z, self.bev_h, self.bev_w), device=voxel_queries.device).to(dtype)).to(dtype)
        #
        # bev_pos_self_attn = self.positional_encoding(
        #     torch.zeros((bs, 512, 512), device=voxel_queries.device).to(dtype)).to(dtype)

        if voxel_feat is not None:
            # ablation study needed
            voxel_feat = voxel_feat.flatten(2).permute(2, 0, 1)  # xyz bs dim
            voxel_queries = voxel_queries + voxel_feat

        # # Load query proposals
        # proposal =  img_metas[0]['proposal'].reshape(self.bev_h, self.bev_w, self.bev_z)
        # unmasked_idx = np.asarray(np.where(proposal.reshape(-1)>0)).astype(np.int32)
        # masked_idx = np.asarray(np.where(proposal.reshape(-1)==0)).astype(np.int32)
        # vox_coords, ref_3d = self.get_ref_3d()

        # load voxel anchor
        # 1 batch supported currently
        voxel_anchor = voxel_anchor[0].cpu().numpy()
        unmasked_idx = np.asarray(np.where(voxel_anchor.reshape(-1)>0)).astype(np.int32)
        # masked_idx = np.asarray(np.where(voxel_anchor.reshape(-1)==0)).astype(np.int32)
        vox_coords, ref_3d = self.get_ref_3d()

        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.transformer.refine_vox_feat(
            mlvl_feats,
            voxel_queries,
            self.bev_h,
            self.bev_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            cam_params=cam_params,
            img_metas=img_metas,
            prev_bev=None,
        )

        # Complete voxel features by adding mask tokens
        voxel_feat = voxel_feat.permute(1, 0, 2)[0]  #
        voxel_feat[vox_coords[unmasked_idx[0], 3], :] = seed_feats[0]
        voxel_feat = voxel_feat.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims)
        voxel_feat = voxel_feat.permute(3, 0, 1, 2).unsqueeze(0)  # B C H W Z
        voxel_feat = voxel_feat.flatten(2).permute(0, 2, 1)

        ref_3d[..., 0:1] = ref_3d[..., 0:1] * \
                          (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        ref_3d[..., 1:2] = ref_3d[..., 1:2] * \
                          (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        ref_3d[..., 2:3] = ref_3d[..., 2:3] * \
                          (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        ref_3d = torch.from_numpy(ref_3d).to('cuda').to(dtype)
        ref_3d = ref_3d.unsqueeze(0)

        voxel_feat = torch.cat([ref_3d, voxel_feat], dim=2).permute(0, 2, 1)

        return voxel_feat
    
    def get_ref_3d(self):
        """Get reference points in 3D. this function is for semantickitti
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
            ref_3d (Array): 3D reference points
        """
        scene_size = self.scene_size
        vox_origin = np.array([self.pc_range[:3]])
        voxel_size = self.real_h / self.bev_h

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([yv.reshape(1,-1), xv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        # Normalize the voxels centroids in lidar cooridnates
        ref_3d = np.concatenate([(yv.reshape(1,-1)+0.5)/self.bev_w, (xv.reshape(1,-1)+0.5)/self.bev_h, (zv.reshape(1,-1)+0.5)/self.bev_z,], axis=0).astype(np.float64).T 

        return vox_coords, ref_3d
