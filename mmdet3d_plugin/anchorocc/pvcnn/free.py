import numpy as np
import torch
import torch.nn as nn


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Not that this should work just as well for 
    continuous values as for discrete values.
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb

ts = torch.tensor([1]).to('cuda')
teb = get_timestep_embedding(128, ts, 'cuda')

print(teb)