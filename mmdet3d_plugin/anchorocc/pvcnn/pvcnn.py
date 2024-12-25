import numpy as np
import torch
import torch.nn as nn

from .modules import Attention
from .pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules
from .pvcnn_utils import get_timestep_embedding

from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmdet.models import NECKS

from projects.mmdet3d_plugin.anchorocc.utils.ssc_loss import CE_ssc_loss


@NECKS.register_module()
class PVCNN2Base(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        embed_dim: int, 
        use_att: bool = True, 
        dropout: float = 0.1,
        extra_feature_channels: int = 3, 
        width_multiplier: int = 1, 
        voxel_resolution_multiplier: int = 1,
        bev_h=None,
        bev_w=None,
        bev_z=None
    ):
        super().__init__()
        assert extra_feature_channels >= 0
        self.out_dim = num_classes
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.width_multiplier = width_multiplier
        
        self.in_channels = extra_feature_channels + embed_dim

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z

        # Create PointNet-2 model
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, 
            extra_feature_channels=extra_feature_channels, 
            with_se=True, 
            embed_dim=embed_dim,
            use_att=use_att, 
            dropout=dropout,
            width_multiplier=width_multiplier, 
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        # Additional global attention module
        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # Only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, 
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=embed_dim, 
            use_att=use_att, 
            dropout=dropout, 
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)

        # Create MLP layers
        self.channels_fp_features = channels_fp_features
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features, 
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True, 
            dim=2, 
            width_multiplier=width_multiplier
        )
        self.classifier = nn.Sequential(*layers)

        # Time embedding function
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(self.out_dim),
        #     nn.Linear(self.out_dim, 20),
        # )

        # self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, inputs: torch.Tensor, t: torch.Tensor):
        """
        The inputs have size (B, 3 + S, N), where S is the number of additional
        feature channels and N is the number of points. The timesteps t can be either 
        continuous or discrete. This model has a sort of U-Net-like structure I think, 
        which is why it first goes down and then up in terms of resolution (?)
        """

        # Embed timesteps
        t_emb = get_timestep_embedding(self.embed_dim, t, inputs.device).float()
        t_emb = self.embedf(t_emb)[:, :, None].expand(-1, -1, inputs.shape[-1])

        # Separate input coordinates and features
        coords = inputs[:, :3, :].contiguous()  # (B, 3, N)
        features = inputs[:, 3:, :].contiguous()  # (B, 3 + S, N)
        
        # Downscaling layers
        coords_list = []
        in_features_list = []
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, t_emb = sa_blocks((features, coords, t_emb))
            else:
                features, coords, t_emb = sa_blocks((torch.cat([features, t_emb], dim=1), coords, t_emb))
        
        # Replace the input features 
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        
        # Apply global attention layer
        if self.global_att is not None:
            features = self.global_att(features)
        
        # Upscaling layers
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            features, coords, t_emb = fp_blocks(
                (  # this is a tuple because of nn.Sequential
                    coords_list[-1 - fp_idx],  # reverse coords list from above
                    coords,  # original point coordinates
                    torch.cat([features, t_emb], dim=1),  # keep concatenating upsampled features with timesteps
                    in_features_list[-1 - fp_idx],  # reverse features list from above
                    t_emb  # original timestep embedding
                )
            )
        # Output MLP layers
        voxel_feat = self.classifier(features)
        voxel_feat = voxel_feat.reshape(1, self.out_dim, self.bev_h, self.bev_w, self.bev_z)

        # out = self.up_scale_2(voxel_feat)
        # out = self.up_scale_2(out)
        # _, feat_dim, w, l, h = out.shape
        # out = out.squeeze().permute(1, 2, 3, 0).reshape(-1, feat_dim)
        # out = self.mlp_head(out)
        # out = out.reshape(w, l, h, 20).permute(3, 0, 1, 2).unsqueeze(0)

        # return voxel_feat, out
        return voxel_feat

@NECKS.register_module()
class PVCNN2(PVCNN2Base):
    # sa_blocks
    # (out_channels, num_blocks, voxel_resolution), (num_centers, radius, num_neighbors, out_channels)
    sa_blocks = [
        ((32, 2, 50), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 25), (256, 0.2, 32, (64, 128))),
        ((128, 3, 16), (64, 0.3, 32, (128, 256))),
        (None, (16, 0.6, 32, (256, 256, 512))),
    ]
    # sa_blocks = [
    #     ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
    #     ((64, 3, 16), (256, 0.2, 32, (64, 128))),
    #     ((128, 3, 8), (64, 0.4, 32, (128, 256))),
    #     (None, (16, 0.8, 32, (256, 256, 512))),
    # ]

    # fp_blocks
    # (out_channels), (out_channels, num_blocks, voxel_resolution)
    fp_blocks = [
        ((256, 256), (256, 3, 16)),
        ((256, 256), (256, 3, 16)),
        ((256, 128), (128, 2, 25)),
        ((128, 128, 64), (64, 2, 50)),
    ]
    # fp_blocks = [
    #     ((256, 256), (256, 3, 8)),
    #     ((256, 256), (256, 3, 8)),
    #     ((256, 128), (128, 2, 16)),
    #     ((128, 128, 64), (64, 2, 32)),
    # ]

    def __init__(self, num_classes, embed_dim, use_att=True, dropout=0.1, extra_feature_channels=3, 
                 width_multiplier=1, voxel_resolution_multiplier=1, bev_h=None, bev_w=None, bev_z=None):
        super().__init__(
            num_classes=num_classes, embed_dim=embed_dim, use_att=use_att,
            dropout=dropout, extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
            bev_h=bev_h, bev_w=bev_w, bev_z=bev_z
        )

        # self.up_scale_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def loss(self, denoise_out, gt_occ, mask_cam=None, dataset=None):

        class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557,
                                                   0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716,
                                                   0.786])).to(denoise_out)
        loss_dict = dict()
        loss_noise = CE_ssc_loss(denoise_out, gt_occ, class_weights)
        loss_dict['loss_noise'] = loss_noise

        return loss_dict
