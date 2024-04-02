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

#sys.path.append(os.path.abspath("/home/moksyasha/Projects/SkyScale/MoksVSR/models/moksvsr"))
from models.moksvsr.MoksVSR import MoksVSR

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


def main():

    BATCH_SIZE = 1
    EPOCHS = 40
    SHOW_PREDS_EVERY = 10 # every 10 epochs, display the model predictions
    N_CORES = multiprocessing.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_path = '/home/moksyasha/Projects/SkyScale/datasets/REDS4/train_sharp'
    ds = VSRDataset(ds_path)
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    # pbar = tqdm(enumerate(dataloader), total=len(ds)//BATCH_SIZE)
    
    model = MoksVSR()
    for lr, hr in dataloader:
        
        lrh, hrh = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
        f, b = model(lrh)
        print("Shape2: ", f.shape, b.shape)

        f_show, b_show = f.permute(0, 1, 3, 4, 2).detach().numpy(), b.permute(0, 1, 3, 4, 2).detach().numpy()
        fig, ax = plt.subplots(2, 2, figsize=(5, 5))
        ax[0][0].imshow(lr[0][1])
        ax[1][0].imshow(lr[0][2])
        ax[0][1].imshow(f_show[0][0])
        ax[1][1].imshow(b_show[0][1])
        plt.show()
        break


    exit()
    # for idx, (lr_h, hr_h) in pbar:
    #     lr, hr = lr_h[0], hr_h[0]
    #     print(f'LR shape: {lr.shape} | HR Shape: {hr.shape} | Dtype: {lr.dtype}')
    #     print(f'LR shape: {lr_h.shape} | HR Shape: {hr_h.shape} | Dtype: {lr_h.dtype}')
    #     fig, ax = plt.subplots(2, 5, figsize=(10, 5))
    #     for i in range(5):
    #         ax[0][i].imshow(lr[i])
    #         ax[1][i].imshow(hr[i])
    #     plt.show()
    #     break

    exit()
    # ds_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/red_val'
    # ds_val = VSRDataset(ds_path)
    # dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    # show some
    # for lr, hr in dataloader:
    #     lr, hr = lr[0], hr[0]
    #     print(f'LR shape: {lr.shape} | HR Shape: {hr.shape} | Dtype: {lr.dtype}')
    #     fig, ax = plt.subplots(2, 5, figsize=(10, 5))
    #     for i in range(5):
    #         ax[0][i].imshow(lr[i])
    #         ax[1][i].imshow(hr[i])
    #     plt.show()
    #     break
    # model = Generator().to(device)
    # opt = Adam(model.parameters(), lr=2e-4)
    # charbonnier_loss_fn = lambda y_pred, y_true, eps=1e-8: torch.mean(torch.sqrt((y_pred - y_true)**2 + eps**2)) # essentially equal to torch.abs(y_pred-y_true) since eps is small

    # train_model_training_loss_ls = []
    # train_model_training_accuracy_ls = []
    # validation_model_training_loss_ls = []
    # validation_model_training_accuracy_ls = []
    # epochs = []
    # for _ in range(EPOCHS//SHOW_PREDS_EVERY):
        # lr, hr = ds[0]
        # lr = torch.Tensor(lr/255.) # normalize
        # lr = lr.permute(0, 3, 1, 2)[None, :, :, :, :] # THWC -> TCHW -> NTCHW
        
        # sr = gen(lr.to(device)) # predict
        # sr = sr.cpu().detach().numpy() # convert to np array
        # lr = lr.numpy()[0].transpose(0, 2, 3, 1) # NTCHW -> TCHW -> THWC
        # sr = sr[0].transpose(0, 2, 3, 1) # NTCHW -> TCHW -> THWC
        # fig, ax = plt.subplots(3, 5, figsize=(10, 5))
        # for i in range(5):
        #     ax[0][i].imshow(lr[i])
        #     ax[1][i].imshow(sr[i])
        #     ax[2][i].imshow(hr[i])
        # plt.show()
        # plt.clf() # clear figure
        # train(gen, dataloader, opt, SHOW_PREDS_EVERY, charbonnier_loss_fn)
    # losses = 0
    
    # for epoch in range(EPOCHS):
    #     num = 0
    #     losses = 0
    #     pbar = tqdm(enumerate(dataloader), total=len(ds)//BATCH_SIZE)
    #     for idx, (lr, hr) in pbar:
    #         # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
    #         lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
    #         lr, hr = lr.to(device), hr.to(device)
    #         num += 1
    #         model.train()
    #         opt.zero_grad()
    #         y_pred = model(lr)
    #         loss = charbonnier_loss_fn(y_pred, hr)
    #         loss.backward()
    #         losses += loss.item()
    #         opt.step()
    #         pbar.set_description(f'Epoch {epoch}, loss: {round(float(loss), 5)}')
    #     a = losses/num
    #     print(a)
    #     epochs.append(epoch)
    #     train_model_training_loss_ls.append(a)
    #     num = 0

    #     # valid
    #     model.eval()  # handle drop-out/batch norm layers
    #     losses = 0
    #     num = 0
    #     with torch.no_grad():
    #         pbar = tqdm(enumerate(dataloader_val), total=len(ds_val)//BATCH_SIZE)
    #         for idx, (lr, hr) in pbar:
    #             # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
    #             lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
    #             lr, hr = lr.to(device), hr.to(device)
    #             y_pred = model(lr)
    #             loss = charbonnier_loss_fn(y_pred, hr)
    #             losses += loss.item()
    #             num += 1
    #     a = losses/num
    #     print(a)
    #     validation_model_training_loss_ls.append(a)

    #     if epoch % 5 == 0:
    #         torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': opt.state_dict(),
    #         'epoch': epoch
    #         }, "val_loss_basic_chkp_" + str(epoch) + ".pt")
    
    # print("train: ", train_model_training_loss_ls)
    # print("val: ", validation_model_training_loss_ls)

    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, train_model_training_loss_ls, label='Training Loss', color='blue', marker='o')
    # plt.plot(epochs, validation_model_training_loss_ls, label='Validation Loss', color='red', marker='o')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


def inference():
    ds_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/red'
    ds = VSRDataset(ds_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Generator().to(device)
    checkpoint = torch.load("/home/moksyasha/Projects/SkyScale/decoding_testing/basic_chkp_" + str(20) + ".pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    lr, hr = ds[0]
    print(torch.Tensor(lr).permute(0, 3, 1, 2)[None, :, :, :, :].shape)
    sr = (model(torch.Tensor(lr/255.).permute(0, 3, 1, 2)[None, :, :, :, :].to(device)).cpu().detach().numpy()[0].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    print(f'LR shape: {lr.shape} | HR Shape: {hr.shape}')
    fig, ax = plt.subplots(3, 5, figsize=(10, 5))
    for i in range(5):
        ax[0][i].imshow(lr[i])
        ax[1][i].imshow(sr[i])
        ax[2][i].imshow(hr[i])
    plt.show()

if __name__ == '__main__':
    main()
    #inference()
