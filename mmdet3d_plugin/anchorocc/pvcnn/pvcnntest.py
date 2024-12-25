from pvcnn import PVCNN2


import torch
import numpy as np


def get_ref_3d():
    """Get reference points in 3D.
    Args:
        self.real_h, self.bev_h
    Returns:
        vox_coords (Array): Voxel indices
        ref_3d (Array): 3D reference points
    """
    scene_size = [51.2, 51.2, 6.4]
    vox_origin = np.array([0, -25.6, -2])
    voxel_size = 51.2 / 128

    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = vox_origin
    vol_bnds[:, 1] = vox_origin + np.array(scene_size)

    # Compute the voxels index in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:, 1] - vol_bnds[:, 0]) / voxel_size).copy(order='C').astype(int)
    idx = np.array([range(vol_dim[0] * vol_dim[1] * vol_dim[2])])
    xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
    vox_coords = np.concatenate([xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1), idx], axis=0).astype(int).T

    # Normalize the voxels centroids in lidar cooridnates
    ref_3d = np.concatenate([(xv.reshape(1, -1) + 0.5) / 128, (yv.reshape(1, -1) + 0.5) / 128,
                             (zv.reshape(1, -1) + 0.5) / 16, ], axis=0).astype(np.float64).T

    return vox_coords, ref_3d


voxcoords, ref3d = get_ref_3d()

pc_range = [0, -25.6, -2, 51.2, 25.6, 4.4]

ref3d[..., 0:1] = ref3d[..., 0:1] * \
                (pc_range[3] - pc_range[0]) + pc_range[0]
ref3d[..., 1:2] = ref3d[..., 1:2] * \
    (pc_range[4] - pc_range[1]) + pc_range[1]
ref3d[..., 2:3] = ref3d[..., 2:3] * \
    (pc_range[5] - pc_range[2]) + pc_range[2]


# print(ref3d)


a = torch.rand(1, 128, 128, 128, 16).flatten(2).to('cuda')
dtype = a.dtype

ref3d = torch.from_numpy(ref3d).to('cuda').to(dtype)
ref3d = ref3d.unsqueeze(0)

a = a.permute(0, 2, 1)

feat = torch.cat([ref3d, a], dim=2).permute(0, 2, 1)

# this verison does not treat 3dim xyz as computable features
# feat tensor has a total 128 + 3 dims
denoiser = PVCNN2(
    embed_dim=128,
    num_classes=18,
    extra_feature_channels=128,
    dropout=0.1, width_multiplier=1,
    voxel_resolution_multiplier=1
).to('cuda')

t = torch.tensor([1]).to('cuda')
out = denoiser(feat, t)

print(out.shape)

