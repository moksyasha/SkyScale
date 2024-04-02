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

class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        spynet_module = lambda: nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3)
            )  

        self.basic_module = nn.ModuleList(
            [spynet_module() for _ in range(6)])

        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def compute_flow(self, ref, supp): # ref = reference image, supp = supporting image
        N, _, H, W = ref.shape
        
        # normalize reference and supporting frames
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]
        
        # generate downscaled images
        for _ in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2))
        ref, supp = ref[::-1], supp[::-1]
        
        # compute and refine flows
        flow = ref[0].new_zeros(N, 2, H // 32, W // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                flow_residue = self.basic_module[level](
                    torch.cat([
                        ref[level],
                        flow_warp(supp[level], flow_up, padding_mode='border'),
                        flow_up
                    ], dim=1))
                
                # add the residue to the upsampled flow
                flow = flow_up + flow_residue
        
        return flow
    
    def forward(self, ref, supp): # upscale ref and supp to be processed, do the processing, rescale the flow to original resolution
        # upsize to a multiple of 32
        H, W = ref.shape[2:4]
        w_up = W if (W % 32) == 0 else 32 * (W // 32 + 1)
        h_up = H if (H % 32) == 0 else 32 * (H // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode='bilinear', align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(self.compute_flow(ref, supp), size=(H, W), mode='bilinear', align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(W) / float(w_up)
        flow[:, 1, :, :] *= float(H) / float(h_up)

        return flow


class MoksVSR(nn.Module):
    def __init__(self, nb=30, mid_channel=64):
        super().__init__()
        self.num_blocks = nb
        self.spynet = SPyNet()

    def compute_flows(self, lrs):
        N, T, C, H, W = lrs.shape
        x_start = lrs[:, :-1, :, :, :].reshape(-1, C, H, W) # reshape from (N, T, C, H, W) into (N*T, C, H, W)
        x_end = lrs[:, 1:, :, :, :].reshape(-1, C, H, W)
        forward_flows = self.spynet(x_start, x_end).view(N, T-1, 2, H, W)
        backward_flows = self.spynet(x_end, x_start).view(N, T-1, 2, H, W)
        
        return forward_flows, backward_flows

    def forward(self, lrs):
        """Forward function.
        Args:
            lrs (Tensor): Input LR sequence with shape (n, t, c, h, w)
            batch of clips, imgs per clip, channel, h, w
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w)
        """

        N, T, C, H, W = lrs.shape
        print(N, T, C, H, W)
        # compute optical flow
        flows_forward, flows_backward = self.compute_flows(lrs) # flow estimation (S step)
        return flows_forward, flows_backward

