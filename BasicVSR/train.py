from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os, glob, multiprocessing, torch
from model.BasicVSR import *
import matplotlib.pyplot as plt


class VSRDataset(Dataset):
    def __init__(self, path, imgs_per_clip=15):
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
        self.lr_path = f'{path}/lr'
        self.hr_path = f'{path}/hr'
        self.imgs_per_clip = imgs_per_clip
        self.lr_folders = sorted(glob.glob(f'{self.lr_path}/*') )
        self.hr_folders = sorted(glob.glob(f'{self.hr_path}/*'))
        self.clips_per_folder = len(glob.glob(f'{self.lr_folders[0]}/*')) // imgs_per_clip
        self.num_clips = len(self.lr_folders) * self.clips_per_folder
    
    def __len__(self):
        return self.num_clips
    
    def __getitem__(self, idx):
        '''
        Returns a np.array of an input video that is of shape
        (T, H, W, 3), where T = imgs per clip, H/W = height/width,
        and 3 = channels (3 for RGB images). Note that the video
        pixel values will be between [0, 255], not [0, 1).
        '''
        folder_idx, clip_idx = idx // self.clips_per_folder, idx % self.clips_per_folder
        s_i, e_i = self.imgs_per_clip * clip_idx, self.imgs_per_clip * (clip_idx + 1)
        lr_fnames = sorted(glob.glob(f'{self.lr_folders[folder_idx]}/*'))[s_i:e_i]
        hr_fnames = sorted(glob.glob(f'{self.hr_folders[folder_idx]}/*'))[s_i:e_i]
        
        for i, (lr_fname, hr_fname) in enumerate(zip(lr_fnames, hr_fnames)):
            lr_img = np.array(Image.open(lr_fname))
            hr_img = np.array(Image.open(hr_fname))
            if i == 0: # instantiate return LR and HR arrays if loading the first image of the batch
                lr_res_y, lr_res_x, _c = lr_img.shape
                hr_res_y, hr_res_x, _c = hr_img.shape
                lr = np.zeros((self.imgs_per_clip, lr_res_y, lr_res_x, 3), dtype=np.uint8)
                hr = np.zeros((self.imgs_per_clip, hr_res_y, hr_res_x, 3), dtype=np.uint8)
            lr[i], hr[i] = lr_img, hr_img
        
        return lr, hr


def train(model, dataloader, opt, epochs, loss_fn):
    '''Trains a VSR model using data provided by a dataloader.
    Args:
    model (nn.Module): the model to be trained.
    dataloader (DataLoader): the data structure which
        provides the model with input/output training data.
    epochs (int): the number of times to reuse the training dataset
        to train the model:
    loss_fn (function): the loss function to minimize.
    '''


def main():

    BATCH_SIZE = 8
    EPOCHS = 40
    SHOW_PREDS_EVERY = 10 # every 10 epochs, display the model predictions
    N_CORES = multiprocessing.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/red'
    ds = VSRDataset(ds_path)
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    ds_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/red_val'
    ds_val = VSRDataset(ds_path)
    dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

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
    model = Generator().to(device)
    opt = Adam(model.parameters(), lr=2e-4)
    charbonnier_loss_fn = lambda y_pred, y_true, eps=1e-8: torch.mean(torch.sqrt((y_pred - y_true)**2 + eps**2)) # essentially equal to torch.abs(y_pred-y_true) since eps is small

    train_model_training_loss_ls = []
    train_model_training_accuracy_ls = []
    validation_model_training_loss_ls = []
    validation_model_training_accuracy_ls = []
    epochs = []
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
    losses = 0
    
    for epoch in range(EPOCHS):
        num = 0
        losses = 0
        pbar = tqdm(enumerate(dataloader), total=len(ds)//BATCH_SIZE)
        for idx, (lr, hr) in pbar:
            # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
            lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
            lr, hr = lr.to(device), hr.to(device)
            num+=1
            model.train()
            opt.zero_grad()
            y_pred = model(lr)
            loss = charbonnier_loss_fn(y_pred, hr)
            loss.backward()
            losses += loss.item()
            opt.step()
            pbar.set_description(f'Epoch {epoch}, loss: {round(float(loss), 5)}')
        a = losses/num
        print(a)
        epochs.append(epoch)
        train_model_training_loss_ls.append(a)
        num = 0

        # valid
        model.eval()  # handle drop-out/batch norm layers
        losses = 0
        num = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader_val), total=len(ds_val)//BATCH_SIZE)
            for idx, (lr, hr) in pbar:
                # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
                lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
                lr, hr = lr.to(device), hr.to(device)
                y_pred = model(lr)
                loss = charbonnier_loss_fn(y_pred, hr)
                losses += loss.item()
                num += 1
        a = losses/num
        print(a)
        validation_model_training_loss_ls.append(a)

        if epoch % 5 == 0:
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch
            }, "val_loss_basic_chkp_" + str(epoch) + ".pt")
    
    print("train: ", train_model_training_loss_ls)
    print("val: ", validation_model_training_loss_ls)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_model_training_loss_ls, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, validation_model_training_loss_ls, label='Validation Loss', color='red', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


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
