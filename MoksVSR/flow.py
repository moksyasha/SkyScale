import os
import time
import numpy as np
import torch
from models.flow.FastFlowNet_v2 import FastFlowNet
#from models.moksvsr.BasicVSR import SPyNet
from PIL import Image
import torch.nn.functional as F
from flow_vis import flow_to_color
import matplotlib.pyplot as plt
import cv2
from torch import nn

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

spynet_module = lambda: nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3), nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=7, stride=1, padding=3)
            )  


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

def main():
    div_flow = 20.0
    div_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img1_path = "/media/moksyasha/linux_data/img1.jpg"
    img2_path = "/media/moksyasha/linux_data/img2.jpg"
    img1 = torch.tensor(cv2.imread(img1_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    img2 = torch.tensor(cv2.imread(img2_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    #img1, img2, _ = centralize(img1, img2)
    # 3 480 854
    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))

    if height % div_size != 0 or width % div_size != 0:
        input_size = (
            int(div_size * np.ceil(height / div_size)), 
            int(div_size * np.ceil(width / div_size))
        )
        img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
        img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    else:
        input_size = orig_size

    # ([1, 6, 448, 1024])
    images = torch.cat((img1, img2), 1).cuda()
    model_fast = FastFlowNet().cuda().eval()
    model_fast.load_state_dict(torch.load('/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/fastflownet_ft_sintel.pth'))
    model_spy = SPyNet().cuda().eval()
    img1 = img1.to(device)
    img2 = img2.to(device)
    #Spynet
    output = model_spy(img1, img2)
    start = time.time()
    output = model_spy(img1, img2)
    print("Spynet output: ", output.shape)  
    end = time.time()
    print("SPYNET The time of execution:",
         (end-start) * 10**3, "ms")

    ## fastflownet
    output = model_fast(images)
    start = time.time()
    output = model_fast(images)
    print("output: ", output.shape) 
    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
    print("flow: ", flow.shape)  
    end = time.time()
    print("FASTFLOW The time of execution:",
         (end-start) * 10**3, "ms")

    if input_size != orig_size:
        scale_h = orig_size[0] / input_size[0]
        scale_w = orig_size[1] / input_size[1]
        flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
        flow[:, 0, :, :] *= scale_w
        flow[:, 1, :, :] *= scale_h

    flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()

    flow_color = flow_to_color(flow, convert_to_bgr=True)
    
    # cv2.imwrite('/home/moksyasha/Projects/SkyScale/MoksVSR/testing_flow/flow1.png', flow_color1)
    # cv2.imwrite('/home/moksyasha/Projects/SkyScale/MoksVSR/testing_flow/flow2.png', flow_color2)

    imgplot = plt.imshow(flow_color)
    plt.show()

if __name__ == '__main__':
    main()