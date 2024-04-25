import os
import time
import numpy as np
import torch
from models.flow.FastFlowNet_v2 import FastFlowNet
from models.moksvsr.BasicVSR import SPyNet
from PIL import Image
import torch.nn.functional as F
from flow_vis import flow_to_color
import matplotlib.pyplot as plt
import cv2

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


def main():
    div_flow = 20.0
    div_size = 64
    img1_path = "/media/moksyasha/linux_data/11.png"
    img2_path = "/media/moksyasha/linux_data/12.png"
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

    # input_t = torch.randn(1, 6, 448, 1024).cuda()
    a = model_spy(img1.cuda(), img2.cuda())
    a = model_spy(img1.cuda(), img2.cuda())
    a = model_spy(img1.cuda(), img2.cuda())
    #a = model_spy(img1.cuda(), img2.cuda())
    # print(a.shape)

    start = time.time()
    print("input: ", images.shape) 
    output = model_spy(img1.cuda(), img2.cuda())
    #output = model_spy(img1.cuda(), img2.cuda())
    print("output: ", output.shape) 
    flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
    print("flow: ", flow.shape)  
    end = time.time()
    print("The time of execution:",
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