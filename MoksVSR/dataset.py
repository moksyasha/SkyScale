from PIL import Image
from torch.utils.data import Dataset
import os
import glob
import numpy as np

class RedsDataset(Dataset):
    def __init__(self, path, imgs_per_clip=10):
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
            represents an input video. Default is 20,
            meaning each input video will consist of
            20 consecutive frames.
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


class VidDataset(Dataset):
    def __init__(self, path, imgs_per_clip=10):
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
            represents an input video. Default is 20,
            meaning each input video will consist of
            20 consecutive frames.
        '''
        self.imgs_per_clip = imgs_per_clip
        self.hr_path = f'{path}'
        self.hr_folders = sorted(glob.glob(f'{self.hr_path}/*'))
        self.clips_per_folder = len(glob.glob(f'{self.hr_folders[0]}/*')) // imgs_per_clip
        self.num_clips = len(self.hr_folders) * self.clips_per_folder # 1200
    
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
        
        hr_names = sorted(glob.glob(f'{self.hr_folders[folder_index]}/*'))[img_start_index : img_end_index]
        
        for i, hr_path in enumerate(hr_names):
            hr_img = np.array(Image.open(hr_path))
            if i == 0: # first create empty array
                hr_y, hr_x, chan = hr_img.shape
                hr_out = np.zeros((self.imgs_per_clip, hr_y, hr_x, chan), dtype=np.uint8)
            hr_out[i] = hr_img
        
        return hr_out

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