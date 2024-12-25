_num_epochs_ = 40
_samples_per_gpu_ = 1
_dataset_ = 'kitti'


_base_ = [
    '../_base_/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

sync_bn = True
plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
camera_used = ['left']
_num_cams_ = 1

# 20 classes with unlabeled
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign',
]
_num_class_ = len(class_names)

point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
scene_size = [point_cloud_range[3]-point_cloud_range[0],
              point_cloud_range[4] - point_cloud_range[1],
              point_cloud_range[5] - point_cloud_range[2]]
# downsample ratio in [x, y, z] when generating 3D volumes
lss_downsample = [2, 2, 2]
volume_h = int(occ_size[0] / lss_downsample[0])
volume_w = int(occ_size[1] / lss_downsample[1])
volume_z = int(occ_size[2] / lss_downsample[2])
volume_size = [volume_h, volume_w, volume_z]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]  # 0.2
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]  # 0.2
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]  # 0.2
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config = {
    'input_size': (384, 1280),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'x': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'y': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'z': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'depth': [1.0, 58.0, 0.5],
}

_dim_ = 128
_num_levels_ = 2
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_layers_cross_ = 3
_num_points_cross_ = 8

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='AnchorOcc',
    dataset=_dataset_,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        out_indices=(2, 3),
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    mlvl_neck=dict(
        type='SECONDFPN',
        in_channels=[_dim_, _dim_],
        upsample_strides=[1, 2],
        out_channels=[_dim_, _dim_]),
    img_view_transformer=dict(
        type='LSSViewTransformerAnchorOcc',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=int(_num_levels_ * _dim_),
        out_channels=_dim_,
        bev_channels=int(volume_z * _dim_),
        bev_h=volume_h,
        bev_w=volume_w,
        sid=False,
        collapse_z=False,
        loss_depth_weight=1.,
        downsample=16),
    anchor_head=dict(
        type='AnchorOccHead',
        bev_h=volume_h,
        bev_w=volume_w,
        bev_z=volume_z,
        scene_size=scene_size,
        pc_range=point_cloud_range,
        embed_dims=_dim_,
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=512,
            col_num_embed=512,
        ),
        transformer=dict(
            type='PerceptionTransformer',
            num_feature_levels=_num_levels_,
            num_cams=_num_cams_,
            embed_dims=_dim_,
            encoder=dict(
                type='AnchorOccEncoder',
                dataset=_dataset_,
                grid_config=grid_config,
                num_layers=_num_layers_cross_,
                pc_range=point_cloud_range,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='AnchorOccLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformCrossAttention',
                            pc_range=point_cloud_range,
                            num_cams=_num_cams_,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_cross_,
                                num_levels=_num_levels_),
                            embed_dims=_dim_,
                        )],
                    ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
            ),),),
    denoise_neck=dict(
        type='PVCNN2',
        bev_h=volume_h,
        bev_w=volume_w,
        bev_z=volume_z,
        embed_dim=128,
        num_classes=64,
        extra_feature_channels=128,
        dropout=0.1,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ),
    occupancy_head=dict(
        type='OccHead',
        in_channels=_dim_//2,
        out_channels=_dim_//2,
        num_class=_num_class_,
        volume_size=volume_size,
        occ_size=occ_size,
        CE_ssc_loss=True,
        sem_scal_loss=True,
        geo_scal_loss=True,
        dice_loss=True,        
    ),
    pts_bbox_head=None,
)

dataset_type = 'CustomSemanticKITTILssDataset'
data_root = 'data/SemanticKITTI'
ann_file = 'data/SemanticKITTI/labels'

bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0,
    )

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=True,
            data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti'),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf, 
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size', 'img_shape']),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=False, 
         data_config=data_config, img_norm_cfg=img_norm_cfg),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size', 'img_shape', 'sequence', 'frame_id', 'raw_img']),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    split='test',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
)

data = dict(
    dataset=_dataset_,
    samples_per_gpu=_samples_per_gpu_,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        split='train',
        camera_used=camera_used,
        occ_size=occ_size,
        pc_range=point_cloud_range,
    ),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)


optimizer = dict(
   type='AdamW',
   lr=1e-4,
   weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 3,
   min_lr_ratio=1e-3)

checkpoint_config = dict(max_keep_ckpts=1, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=_num_epochs_)

# load_from = 'ckpts/deno.pth'

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='semkitti_SSC_mIoU',
    rule='greater',
)
