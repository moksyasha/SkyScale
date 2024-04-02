#https://github.com/open-mmlab/mmediting/blob/a852622b837274ff845b385eb895d540bfe6981e/mmedit/models/backbones/sr_backbones/basicvsr_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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