import torch.nn.functional as F


def interpolate_latents(x, size=None, scale_factor=None, mode="linear", type=None):
    if x.dim() != 3:
        raise ValueError("HiFlow-SAE interpolation expects [B, T, C] latents")
    x = x.transpose(1, 2)
    x = F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=False)
    return x.transpose(1, 2)
