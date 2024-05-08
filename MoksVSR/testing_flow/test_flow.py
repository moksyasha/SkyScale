import torchvision.transforms as T
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os, glob, multiprocessing, torch
#from model.BasicVSR import *
import matplotlib.pyplot as plt
import time

import sys
sys.path.append(os.path.abspath("/home/moksyasha/Projects/SkyScale/MoksVSR/"))
#from SpyNetflow import SPyNet

from modelsflow.FastFlowNet_v2 import FastFlowNet
from flow_vis import flow_to_color

class VSRDataset(Dataset):
    def __init__(self, path, imgs_per_clip=20):
        '''
        Args:
        lr_path (str): Represents a path that contains a set of folders,
            where each folder contains a sequence of
            consecutive LR frames of a video.
        hr_path (str): See lr_path, but each folder
            contains HR frames. The folder and image names,
            when sorted, should be a 1:1 match with the LR frames
            (i.e. the third image in the second folder of the lr_path
            should be the LR image ofthe third image
            in the second folder of the hr_path).
        imgs_per_clip (int): The number of images that
            represents an input video. Default is 15,
            meaning each input video will consist of
            15 consecutive frames.
        '''
        self.lr_path = f'{path}_bicubic/'
        self.hr_path = f'{path}'
        self.imgs_per_clip = imgs_per_clip
        self.lr_folders = sorted(glob.glob(f'{self.lr_path}/*'))
        self.hr_folders = sorted(glob.glob(f'{self.hr_path}/*'))
        self.clips_per_folder = len(glob.glob(f'{self.lr_folders[0]}/*')) // imgs_per_clip
        self.num_clips = len(self.lr_folders) * self.clips_per_folder # 1200
    
    def __len__(self):
        return self.num_clips
    
    def __getitem__(self, idx):
        '''
        Returns a np.array of an input video that is of shape
        (T, H, W, 3), where T = imgs per clip, H/W = height/width,
        and 3 = channels (3 for RGB images). Note that the video
        pixel values will be between [0, 255], not [0, 1).
        '''
        folder_index = idx // self.clips_per_folder
        clip_index = idx % self.clips_per_folder
        img_start_index = self.imgs_per_clip * clip_index
        img_end_index = self.imgs_per_clip * (clip_index + 1) # 0-19, 20-39, ...
        lr_names = sorted(glob.glob(f'{self.lr_folders[folder_index]}/*'))[img_start_index : img_end_index]
        hr_names = sorted(glob.glob(f'{self.hr_folders[folder_index]}/*'))[img_start_index : img_end_index]
        
        for i, (lr_path, hr_path) in enumerate(zip(lr_names, hr_names)):
            lr_img = np.array(Image.open(lr_path))
            hr_img = np.array(Image.open(hr_path))
            if i == 0: # first create empty array
                lr_y, lr_x, chan = lr_img.shape
                hr_y, hr_x, chan = hr_img.shape
                lr_out = np.zeros((self.imgs_per_clip, lr_y, lr_x, chan), dtype=np.uint8)
                hr_out = np.zeros((self.imgs_per_clip, hr_y, hr_x, chan), dtype=np.uint8)
            lr_out[i], hr_out[i] = lr_img, hr_img
        
        return lr_out, hr_out

        # folder_idx, clip_idx = idx // self.clips_per_folder, idx % self.clips_per_folder
        # s_i, e_i = self.imgs_per_clip * clip_idx, self.imgs_per_clip * (clip_idx + 1)
        # lr_fnames = sorted(glob.glob(f'{self.lr_folders[folder_idx]}/*'))[s_i:e_i]
        # hr_fnames = sorted(glob.glob(f'{self.hr_folders[folder_idx]}/*'))[s_i:e_i]
        
        # for i, (lr_fname, hr_fname) in enumerate(zip(lr_fnames, hr_fnames)):
        #     lr_img = np.array(Image.open(lr_fname))
        #     hr_img = np.array(Image.open(hr_fname))
        #     if i == 0: # instantiate return LR and HR arrays if loading the first image of the batch
        #         lr_res_y, lr_res_x, _c = lr_img.shape
        #         hr_res_y, hr_res_x, _c = hr_img.shape
        #         lr = np.zeros((self.imgs_per_clip, lr_res_y, lr_res_x, 3), dtype=np.uint8)
        #         hr = np.zeros((self.imgs_per_clip, hr_res_y, hr_res_x, 3), dtype=np.uint8)
        #     lr[i], hr[i] = lr_img, hr_img
        
        # return lr, hr

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def flow_to_color(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def main():

    # BATCH_SIZE = 1
    # EPOCHS = 40
    # SHOW_PREDS_EVERY = 10 # every 10 epochs, display the model predictions
    # N_CORES = multiprocessing.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ds_path = '/media/moksyasha/linux_data/datasets/REDS4/train_sharp'
    # ds = VSRDataset(ds_path)
    # dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    # # pbar = tqdm(enumerate(dataloader), total=len(ds)//BATCH_SIZE)
    
    # lr, hr = ds[0]
    # ref, sup = torch.tensor(lr[0]).permute(2, 0, 1).unsqueeze(0)/255., torch.tensor(lr[2]).permute(2, 0, 1).unsqueeze(0)/255.
    
    img1 = torch.tensor(np.array(Image.open("/media/moksyasha/linux_data/img1.jpg"))).permute(2, 0, 1).unsqueeze(0)/255.
    img2 = torch.tensor(np.array(Image.open("/media/moksyasha/linux_data/img2.jpg"))).permute(2, 0, 1).unsqueeze(0)/255.
    
    print(img1.shape)
    height, width = img1.shape[-2:]
    orig_size = (int(height), int(width))
    print("Orig: ", orig_size)
    #if height % div_size != 0 or width % div_size != 0:
    input_size = (
        int(np.ceil(height / 1)), 
        int(np.ceil(width / 1))
    )
    import torchvision.transforms as transforms
    resize = transforms.Resize(input_size)
    img1 = resize(img1)
    img2 = resize(img2)
    # img1 = F.resize(img1, size=input_size, mode='bilinear', align_corners=False)
    # img2 = F.resize(img2, size=input_size, mode='bilinear', align_corners=False)
    imgplot = plt.imshow(img2.squeeze().permute(1, 2, 0).numpy())
    plt.show()

    images = torch.cat((img1, img2), 0).unsqueeze(0).float().cuda().to(device)
    print("iamges:" , images.shape)
    #########################################33
    # flow_model = SPyNet(None)
    # start = time.time()

    # a = []
    # for i in range(100):
    #     flow = flow_model(torch.tensor(lr_img1).permute(2, 0, 1).unsqueeze(0)/255., torch.tensor(lr_img2).permute(2, 0, 1).unsqueeze(0)/255.).squeeze().permute(1, 2, 0).detach().numpy()
    #     end = time.time()
    #     print("The time of execution:",
    #         (end-start) * 10**3, "ms")
    #     start = end
    # #print(flow.shape)


    # image = flow_to_color(flow, True)
    # print("Image: ", image.shape)
   
    # imgplot = plt.imshow(image)
    # plt.show()
    #########################################33

    import cv2 as cv
    import ptlflow
    from ptlflow.utils import flow_utils
    from ptlflow.utils.io_adapter import IOAdapter

    model = ptlflow.get_model('rpknet', pretrained_ckpt='sintel')
    model = model.to(device)
    # A helper to manage inputs and outputs of the model
    # io_adapter = IOAdapter(model, images[0].shape[:2])

    # inputs is a dict {'images': torch.Tensor}
    # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
    # (1, 2, 3, H, W)
    #inputs = io_adapter.prepare_inputs(images)
    predictions = model({"images": images})
    # Forward the inputs through the model
    start = time.time()
    predictions = model({"images": images})
    end = time.time()
    print("The time of execution:",
        (end-start) * 10**3, "ms")
    # The output is a dict with possibly several keys,
    # but it should always store the optical flow prediction in a key called 'flows'.
    flows = predictions['flows']

    # flows will be a 5D tensor BNCHW.
    # This example should print a shape (1, 1, 2, H, W).
    print(flows.shape)

    # Create an RGB representation of the flow to show it on the screen
    flow_rgb = flow_utils.flow_to_rgb(flows)
    # Make it a numpy array with HWC shape
    flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
    flow_rgb_npy = flow_rgb.detach().cpu().numpy()
    # OpenCV uses BGR format
    flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)

    #flow_bgr_npy = F.interpolate(flow_bgr_npy, size=orig_size, mode='bilinear', align_corners=False)
    

    imgplot = plt.imshow(flow_bgr_npy)
    plt.show()



    

###########################3№№№№№№№№№№№

      
    # div_flow = 20.0
    # div_size = 64

    # def centralize(img1, img2):
    #     b, c, h, w = img1.shape
    #     rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    #     return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

    # model = FastFlowNet().cuda().eval()
    # model.load_state_dict(torch.load('/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/fastflownet_ft_mix.pth'))

    # # img1 = torch.from_numpy(cv2.imread(img1_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    # # img2 = torch.from_numpy(cv2.imread(img2_path)).float().permute(2, 0, 1).unsqueeze(0)/255.0
    # #img1, img2, _ = centralize(img1, img2)

    # img1, img2 = ref, sup
    # height, width = img1.shape[-2:]
    # orig_size = (int(height), int(width))

    # if height % div_size != 0 or width % div_size != 0:
    #     input_size = (
    #         int(div_size * np.ceil(height / div_size)), 
    #         int(div_size * np.ceil(width / div_size))
    #     )
    #     img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
    #     img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
    # else:
    #     input_size = orig_size

    # input_t = torch.cat([img1, img2], 1).cuda()
    # start = time.time()
    # output = model(input_t).data
    # end = time.time()
    # print("The time of execution:", (end-start) * 10**3, "ms")
    # flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

    # if input_size != orig_size:
    #     scale_h = orig_size[0] / input_size[0]
    #     scale_w = orig_size[1] / input_size[1]
    #     flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
    #     flow[:, 0, :, :] *= scale_w
    #     flow[:, 1, :, :] *= scale_h


    # flow = flow[0].cpu().permute(1, 2, 0).numpy()

    # flow_color = flow_to_color(flow, convert_to_bgr=True)

    # imgplot = plt.imshow(flow_color)
    # plt.show()
    # # # cv2.imwrite('./data/flow.png', flow_color)



    exit()







    for lr, hr in dataloader:
        
        
        # lrh, hrh = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
        
        # tfms = T.Compose([
        # T.ToTensor(),
        # T.Normalize(mean=[.485, .406, .456], 
        #             std= [.229, .225, .224])
        # ])

        # model = spynet.SpyNet.from_pretrained('sentinel')
        # model.eval()

        # frame1 = tfms(Image.open('..')).unsqueeze(0)
        # frame2 = tfms(Image.open('..')).unsqueeze(0)

        # flow = model((frame1, frame2))[0]
        # flow = spynet.flow.flow_to_image(flow)
        # Image.fromarray(flow).show()
        # f, b = model(lrh)
        # print("Shape2: ", f.shape, b.shape)

        # f_show, b_show = f.permute(0, 1, 3, 4, 2).detach().numpy(), b.permute(0, 1, 3, 4, 2).detach().numpy()
        # fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        # ax[0][0].imshow(lr[0][1])
        # ax[1][0].imshow(lr[0][2])
        # ax[0][1].imshow(f_show[0][0])
        # ax[1][1].imshow(b_show[0][1])
        # plt.show()
        break

if __name__ == '__main__':
    main()