#https://github.com/open-mmlab/mmediting/blob/3351cdc398dc34813614e021269a92b4ad84da92/mmedit/models/common/flow_warp.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    '''Warp an image or a feature map with optical flow.
    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.
    Returns:
        Tensor: Warped image or feature map.
    '''
    _N, _C, H, W = x.shape
    device = flow.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=device, dtype=x.dtype),
        torch.arange(0, W, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), dim=2)
    grid_flow = flow.permute((0, 2, 3, 1)) + grid # flow shape: NCHW to NHWC
    
    # normalize grid flows to [-1, 1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(W - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(H - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    
    return output