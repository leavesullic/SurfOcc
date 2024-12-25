import torch
import torch.nn.functional as F
import collections 

from mmdet.models import DETECTORS
from mmcv.runner import force_fp32
from mmdet3d.models import builder

from projects.mmdet3d_plugin.utils import fast_hist_crop
from .bevdepth import BEVDepth
from mmdet3d.models.detectors import CenterPoint

import numpy as np
import time
import pdb

@DETECTORS.register_module()
class AnchorOcc(CenterPoint):
    def __init__(self,
                 dataset=None,
                 mlvl_neck=None,
                 img_view_transformer=None,
                 anchor_head=None,
                 denoise_neck=None,
                 occupancy_head=None,
                 **kwargs):
        super(AnchorOcc, self).__init__(**kwargs)
        
        self.dataset = dataset
        
        if mlvl_neck is not None:
            self.mlvl_neck = builder.build_neck(mlvl_neck)
            
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        
        if anchor_head is not None:
            self.anchor_head = builder.build_head(anchor_head)


        if denoise_neck is not None:
            self.denoise_neck = builder.build_neck(denoise_neck)

        if occupancy_head is not None:
            self.occupancy_head = builder.build_head(occupancy_head)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        img_feats = self.img_backbone(imgs)

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        mlvl_feats = []
        for feat in img_feats:
            BN, C, H, W = feat.size()
            mlvl_feats.append(feat.view(B, int(BN / B), C, H, W))

        img_feat = self.mlvl_neck(img_feats)
        if type(img_feat) in [list, tuple]:
            img_feat = img_feat[0]

        _, output_dim, ouput_H, output_W = img_feat.shape
        img_feat = img_feat.view(B, N, output_dim, ouput_H, output_W)
        
        return img_feat, mlvl_feats
    
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
                
        img_feat, mlvl_feats = self.image_encoder(img[0])
        # img_feats = x.clone()
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        

        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        
        voxel_feat, voxel_anchor, depth = self.img_view_transformer([img_feat] + geo_inputs)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        return voxel_feat, voxel_anchor, depth, mlvl_feats


    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        voxel_feat, voxel_anchor, depth, mlvl_feats = self.extract_img_feat(img, img_metas)

        return (voxel_feat, voxel_anchor, mlvl_feats, depth)

    def forward_occ_train(
            self,
            voxel_feat,
            voxel_anchor,
            gt_occ,
            mask_cam,
            img_metas,
            cam_params,
            point_occ,
            img_feats=None,
            points_uv=None,
            **kwargs
        ):

        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()

        
        voxel_feat = self.anchor_head(img_feats, voxel_feat, voxel_anchor, cam_params, img_metas)

        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['anchor_head'].append(t1 - t0)

        t = torch.tensor([1]).to(voxel_feat)
        denoise_voxel_feat = self.denoise_neck(voxel_feat, t)
        # loss_noise = self.denoise_neck.loss(denoise_out, gt_occ, mask_cam, self.dataset)

        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['denoise_neck'].append(t2 - t1)

        outs = self.occupancy_head(denoise_voxel_feat)
        losses_occupancy = self.occupancy_head.loss(outs, gt_occ, mask_cam, self.dataset)

        # losses_occupancy.update(loss_noise)

        if self.record_time:
            torch.cuda.synchronize()
            t3 = time.time()
            self.time_stats['occupancy_head'].append(t3 - t2)

        return losses_occupancy
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            mask_camera=None,
            points_occ=None,
            points_uv=None,
            **kwargs,
        ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        # extract bird-eye-view features from perspective images
        voxel_feat, voxel_anchor, img_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[7], depth)
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
        
        if self.dataset == 'kitti':
            _, rots, trans, intrins, post_rots, post_trans, bda, _, _, lidar2img = img_inputs
            cam_params = rots, trans, intrins, post_rots, post_trans, bda, lidar2img
        elif self.dataset == 'nuscenes':
            _, rots, trans, intrins, post_rots, post_trans, bda, _, _ = img_inputs
            cam_params = rots, trans, intrins, post_rots, post_trans, bda

        losses_occupancy = self.forward_occ_train(voxel_feat, voxel_anchor, gt_occ, mask_camera, img_metas, cam_params,
                        points_occ, img_feats=img_feats, points_uv=points_uv, **kwargs)
        
        losses.update(losses_occupancy)
        
        if self.record_time:
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
                
            print(out_res)
        
        return losses
    
    def forward_occ_test(
            self,
            voxel_feat,
            voxel_anchor,
            cam_params,
            img_metas,
            point_occ,
            img_feats=None,
            points_uv=None,
        ):

        voxel_feat = self.anchor_head(img_feats, voxel_feat, voxel_anchor, cam_params, img_metas)

        t = torch.tensor([1]).to(voxel_feat)
        denoise_voxel_feat = self.denoise_neck(voxel_feat, t)

        outs = self.occupancy_head(denoise_voxel_feat)

        return outs
        
    def forward_test(self,
            img_metas=None,
            img_inputs=None,
            **kwargs,
        ):
        
        return self.simple_test(img_metas, img_inputs, **kwargs)
    
    def simple_test(self, img_metas, img=None, rescale=False, points_occ=None, gt_occ=None, points_uv=None):
        
        voxel_feat, voxel_anchor, img_feats, _ = self.extract_feat(points=None, img=img, img_metas=img_metas)
        
        if self.dataset == 'kitti':
            _, rots, trans, intrins, post_rots, post_trans, bda, _, _, lidar2img = img
            cam_params = rots, trans, intrins, post_rots, post_trans, bda, lidar2img
        elif self.dataset == 'nuscenes':
            _, rots, trans, intrins, post_rots, post_trans, bda, _, _ = img
            cam_params = rots, trans, intrins, post_rots, post_trans, bda
         
        out = self.forward_occ_test(voxel_feat, voxel_anchor, cam_params, img_metas,
                        points_occ, img_feats=img_feats, points_uv=points_uv)

        output_voxels = out["ssc_logit"]
        output = dict()

        output['output_voxels'] = output_voxels
        output['target_voxels'] = gt_occ

        if self.dataset != 'kitti':
            # nuscenes
            out_voxels = output_voxels.permute(0, 2, 3, 4, 1)[0]
            occ = out_voxels.softmax(-1)
            
            # whether convert matrix format to cvpr2023
            occ = occ.permute(3, 2, 0, 1)
            occ = torch.flip(occ, [2])
            occ = torch.rot90(occ, -1, [2, 3])
            occ = occ.permute(2, 3, 1, 0)
            
            occ = occ.argmax(-1)
            occ = occ.unsqueeze(0)
            
            
            
            return occ
        
        return output

