import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE

from .models_mae import MaskedAutoencoderViT

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVMaskedAutoencoder(MaskedAutoencoderViT):
    def __init__(self, **kwargs):
        super(BEVMaskedAutoencoder, self).__init__(**kwargs)

    def get_min_anchor_num(self, bev_anchor_mask):
        B, _, _ = bev_anchor_mask.shape
        anchor_num = []
        for i in range(B):
            anchor_num_i = torch.nonzero(bev_anchor_mask[i]).shape[0]
            anchor_num.append(anchor_num_i)

        min_anchor_num = sorted(anchor_num)[0]
        return min_anchor_num

    def bev_anchor_masking(self, bev_feat, bev_anchor_mask):
        B, L, D = bev_feat.shape
        min_anchor_num = self.get_min_anchor_num(bev_anchor_mask)

        # sort 0 1 in anchor mask for each sample
        ids_shuffle = torch.argsort(bev_anchor_mask.flatten(1), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # for every batch, due to different depth prediction
        # different num of anchors will generate for each sample
        # keep the min anchor num for batch
        ids_keep = ids_shuffle[:, -min_anchor_num:]
        masked_bev_feat = torch.gather(bev_feat, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        return masked_bev_feat, ids_restore

    def forward_encoder(self, x, bev_anchor_mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> anchor num
        x, ids_restore = self.bev_anchor_masking(x, bev_anchor_mask)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore

    def forward(self, bev_img, bev_anchor_mask):
        # B euqals to 1
        B, C, H, W = bev_img.shape
        latent, ids_restore = self.forward_encoder(bev_img, bev_anchor_mask)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        # restore to BEV
        bev_feat = pred.transpose(1, 2).reshape(B, C, H, W)

        return bev_feat
