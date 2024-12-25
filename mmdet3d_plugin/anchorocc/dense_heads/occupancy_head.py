import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from torch.utils.checkpoint import checkpoint as cp
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast
from mmdet3d.models import builder
from mmdet.models.builder import build_loss


from projects.mmdet3d_plugin.anchorocc.utils.ssc_loss import CE_ssc_loss, geo_scal_loss, sem_scal_loss, dice_loss


BIAS = True
def conv3x3x3(in_planes, out_planes, stride=1, use_spase_3dtensor=False):
    Conv3d = nn.Conv3d
    return Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=BIAS)

def conv1x1x1(in_planes, out_planes, stride=1, use_spase_3dtensor=False):
    Conv3d = nn.Conv3d
    return Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=BIAS)

class BasicBlock(BaseModule):
    def __init__(self, in_planes, planes, stride=1, norm_cfg=None):
        super().__init__()
        self.expansion = 1
        self.relu = nn.ReLU(inplace=False)

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]

        self.stride = stride
        
        self.downsample = nn.Sequential(
                    conv1x1x1(in_planes, planes * self.expansion, stride),
                    build_norm_layer(norm_cfg, planes * self.expansion)[1])

    @force_fp32()
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
        

class Res3DBlock(BaseModule):
    def __init__(self, in_channels=128, out_channels=128):
        super().__init__()

        self.norm_cfg = dict(type='BN3d', requires_grad=True)
        self.resnet3d_block = BasicBlock(
            in_channels,
            out_channels,
            norm_cfg=self.norm_cfg
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, voxel_feat):

        out = self.resnet3d_block(voxel_feat)
        
        return out


class ASPP3D(BaseModule):
    """
    ASPP 3D
    Adapt from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, planes, dilations_conv_list=[1, 2, 3]):
        super().__init__()

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_in):

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        return x_in

    

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_class=20,
        volume_size=[64, 64, 8],
        occ_size=[256, 256, 32],
        class_weights=[0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 
                    0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786],
        CE_ssc_loss=True,
        sem_scal_loss=True,
        geo_scal_loss=True,
        dice_loss=True,
        nusc_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            )
    ):
        super(OccHead, self).__init__()

        self.in_dim = in_channels
        self.out_dim = out_channels
        self.n_classes = num_class
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.out_dim),
            nn.Linear(self.out_dim, self.n_classes),
        )

        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.dice_loss = dice_loss
        
        self.nusc_loss = build_loss(nusc_loss)

        self.class_weights = torch.from_numpy(np.array(class_weights))

        scale_times = int(np.math.log2(occ_size[0] // volume_size[0]))
        
        sequence = []
        if scale_times == 0:
            sequence.append(Res3DBlock(self.in_dim, self.out_dim))
            sequence.append(ASPP3D(self.out_dim))
        
        else:
            for _ in range(scale_times - 1):
                sequence.append(Res3DBlock(self.in_dim, self.in_dim))
                sequence.append(nn.ConvTranspose3d(self.in_dim, self.in_dim, kernel_size=4, padding=1, stride=2))
                sequence.append(nn.BatchNorm3d(self.in_dim))
                sequence.append(nn.ReLU(inplace=True))

            sequence.append(Res3DBlock(self.in_dim, self.out_dim))
            sequence.append(nn.ConvTranspose3d(self.out_dim, self.out_dim, kernel_size=4, padding=1, stride=2))
            sequence.append(nn.BatchNorm3d(self.out_dim))
            sequence.append(nn.ReLU(inplace=True))
            sequence.append(ASPP3D(self.out_dim))

        self.sequence = nn.Sequential(*sequence)

    @force_fp32(apply_to=('voxel_feat'))
    def forward(self, voxel_feat):
        out = {}
        voxel_feat = self.sequence(voxel_feat)

        B, feat_dim, H, W, Z = voxel_feat.shape  # 1, _, 256, 256, 32
        voxel_feat = voxel_feat.squeeze().permute(1, 2, 3, 0).reshape(-1, feat_dim)
        ssc_logit = self.mlp_head(voxel_feat)

        out["ssc_logit"] = ssc_logit.reshape(H, W, Z, self.n_classes).permute(3,0,1,2).unsqueeze(0)
        
        return out


    @force_fp32()
    def loss(self, out_dict, target, mask_cam=None, dataset=None):

        ssc_pred = out_dict["ssc_logit"]
        
        ssc_pred[torch.isnan(ssc_pred)] = 0
        ssc_pred[torch.isinf(ssc_pred)] = 0
        assert torch.isnan(ssc_pred).sum().item() == 0
        assert torch.isnan(ssc_pred).sum().item() == 0
        
        loss_dict = dict()
        class_weight = self.class_weights.to(ssc_pred)
        
        if dataset == 'kitti':
        
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
                loss_dict['loss_ssc'] = loss_ssc

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target, empty_idx=0)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            if self.dice_loss:
                loss_dice = dice_loss(ssc_pred, target)
                loss_dict['loss_dice'] = loss_dice

            return loss_dict
        
        else:
            # nuscenes
            assert target.min()>=0 and target.max()<=17
            
            if mask_cam is not None:
                loss_ssc = self.loss_single(target, mask_cam, ssc_pred)
                loss_dict['loss_ssc'] = loss_ssc
            else:
                voxel_semantics = target.reshape(-1)
                occ = ssc_pred.reshape(-1, self.n_classes)
                loss_ssc = self.nusc_loss(occ, voxel_semantics)
                loss_dict['loss_ssc'] = loss_ssc

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target, empty_idx=17)
                loss_dict['loss_geo_scal'] = loss_geo_scal

            if self.dice_loss:
                loss_dice = dice_loss(ssc_pred, target)
                loss_dict['loss_dice'] = loss_dice

            return loss_dict
  
    
    def loss_single(self, voxel_semantics, mask_camera, preds):
        voxel_semantics = voxel_semantics.long()
        
        voxel_semantics = voxel_semantics.reshape(-1)
        preds = preds.reshape(-1, self.n_classes)
        mask_camera = mask_camera.reshape(-1)
        num_total_samples = mask_camera.sum()

        loss_occ = self.nusc_loss(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)

        return loss_occ







            



    
