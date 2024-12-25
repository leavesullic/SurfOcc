#import open3d as o3d
import numpy as np
import yaml, os
import torch
import torch.nn.functional as F
import numba as nb

from PIL import Image
from mmdet.datasets.builder import PIPELINES

import pdb

@PIPELINES.register_module()
class LoadNuscOccupancyAnnotation(object):
    def __init__(
            self,
            data_root='data/occ3d-nus',
            is_train=False,
            is_test_submit=False,
            bda_aug_conf=None,
            unoccupied_id=17,
        ):
        
        self.is_train = is_train
        self.is_test_submit = is_test_submit
        
        self.data_root = data_root
        self.bda_aug_conf = bda_aug_conf
        self.unoccupied_id = unoccupied_id

    
    def sample_3d_augmentation(self):
        """Generate 3d augmentation values based on bda_config."""
        
        # Currently, we only use the flips along three directions. The rotation and scaling are not fully experimented.
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf.get('flip_dz_ratio', 0.0)
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

    def __call__(self, results):
        # for test-submission of nuScenes LiDAR Segmentation 
        if self.is_test_submit:
            imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
            bda_rot = torch.eye(3).float()
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)
            
            pts_filename = results['pts_filename']
            points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
            points_label = np.zeros((points.shape[0], 1)) # placeholder
            lidarseg = np.concatenate([points, points_label], axis=-1)
            results['points_occ'] = torch.from_numpy(lidarseg).float()
            
            return results
        
        if 'occ_gt_path' in results:
            occ_gt_path = results['occ_gt_path']
            occ_gt_path = os.path.join(self.data_root, occ_gt_path)

            occ_labels = np.load(occ_gt_path)
            gt_occ = occ_labels['semantics']
            mask_lidar = occ_labels['mask_lidar']
            mask_camera = occ_labels['mask_camera']
        else:
            gt_occ = np.zeros((200,200,16),dtype=np.uint8)
            mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
            mask_camera = np.zeros((200, 200, 16), dtype=np.uint8)

        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        
        

        if self.is_train:
            rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_3d_augmentation()
            gt_occ, bda_rot = voxel_transform(gt_occ, rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz)
        else:
            bda_rot = torch.eye(3).float()
            
        gt_occ = torch.from_numpy(gt_occ.copy()).long()
      
        results['gt_occ'] = gt_occ.long()
        imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors)

        return results
    

def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz):
    # bird-eye-view rotation
    rotate_degree = rotate_angle
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([
        [rot_cos, -rot_sin, 0, 0],
        [rot_sin, rot_cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    # I @ flip_x @ flip_y
    flip_mat = torch.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([
            [-1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dz:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, -1, 0],
            [0, 0, 0, 1]])
    
    # denorm @ flip_x @ flip_y @ flip_z @ rotation @ normalize
    bda_mat = flip_mat @ rot_mat
    bda_mat = bda_mat[:3, :3]
    
    # apply transformation to the 3D volume, which is tensor of shape [X, Y, Z]
    if voxel_labels is not None:
        voxel_labels = voxel_labels.astype(np.uint8)
        if not np.isclose(rotate_degree, 0):
            '''
            Currently, we use a naive method for 3D rotation because we found the visualization of 
            rotate results with scipy is strange: 
                scipy.ndimage.interpolation.rotate(voxel_labels, rotate_degree, 
                        output=voxel_labels, mode='constant', order=0, 
                        cval=255, axes=(0, 1), reshape=False)
            However, we found further using BEV rotation brings no gains over 3D flips only.
            '''
            voxel_labels = custom_rotate_3d(voxel_labels, rotate_degree)
        
        if flip_dz:
            voxel_labels = voxel_labels[:, :, ::-1]
        
        if flip_dy:
            voxel_labels = voxel_labels[:, ::-1]
        
        if flip_dx:
            voxel_labels = voxel_labels[::-1]
        
        # voxel_labels = torch.from_numpy(voxel_labels.copy()).long()
    
    return voxel_labels, bda_mat

def custom_rotate_3d(voxel_labels, rotate_degree):
    # rotate like images: convert to PIL Image and rotate
    is_tensor = False
    if type(voxel_labels) is torch.Tensor:
        is_tensor = True
        voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    voxel_labels_list = []
    for height_index in range(voxel_labels.shape[-1]):
        bev_labels = voxel_labels[..., height_index]
        bev_labels = Image.fromarray(bev_labels.astype(np.uint8))
        bev_labels = bev_labels.rotate(rotate_degree, resample=Image.Resampling.NEAREST, fillcolor=255)
        bev_labels = np.array(bev_labels)
        voxel_labels_list.append(bev_labels)
    voxel_labels = np.stack(voxel_labels_list, axis=-1)
    
    if is_tensor:
        voxel_labels = torch.from_numpy(voxel_labels).long()
    
    return voxel_labels