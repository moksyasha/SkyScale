import os, glob, multiprocessing, torch
from torch.utils.data import DataLoader
from models.moksvsr.MoksVSR import MoksVSR, MoksPlus, TestMoksVSRDouble
from models.moksvsr.BasicVSR import Generator
from dataset import VidDataset
import argparse
from tqdm import tqdm
import numpy as np
import erqa
import matplotlib.pyplot as plt
import time
import cv2
import torchvision.transforms as T

def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('--config', default="/home/moksyasha/Projects/SkyScale/MoksVSR/config/configs/basicvsr/6464crop_moks__28.py", help='test config file path')
    parser.add_argument('--checkpoint', default="/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/plus_resize64_29.pt", help='checkpoint file')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1
    N_CORES = multiprocessing.cpu_count()

    #ds_test = RedsDataset('/media/moksyasha/linux_data/datasets/REDS4/test_sharp', 20)
    #dataloader_val = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CORES)
    ds_test = VidDataset('/media/moksyasha/linux_data/datasets/Vid4', 10)
    dataloader_val = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CORES)

    model = MoksPlus()
    #model_2 = Generator()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    model = model.to(device)
    #model_2 = model_2.to(device)

    model.eval()  # handle drop-out/batch norm layers
    #model_2.eval()
    losses = 0
    num = 0
    metric = erqa.ERQA()
    erqa_arr = []
    num = 0
    red = 0
    with torch.no_grad():

        if red:
            pbar = tqdm(enumerate(dataloader_val), total=len(ds_test)//BATCH_SIZE)
            for idx, (lr, hr) in pbar:
                # if idx == 5:
                #     break
                # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
                lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 3, 2, 4).cpu().numpy() / 255.
                lr, hr = lr[:, :, ...], hr[:, :, ...]
                #lr = (lr - mean) / std
                # 1 b c h w
                lr = lr.to(device)

                # y pred
                # 1 b c h w
                start = time.time()
                y_pred = model(lr).squeeze().permute(0, 2, 3, 1).cpu().numpy()
                end = time.time()
                print("The time of execution:",
                    (end-start) * 10**3, " ms")
                #print("Shape: ", lr.shape, y_pred.shape)
                # imgplot = plt.imshow(y_pred[1])
                # plt.show()
                for pred in y_pred:
                    cv2.imwrite('/home/moksyasha/Projects/SkyScale/MoksVSR/results/vid_plus_pred/{:06d}.png'.format(num), cv2.cvtColor(255*pred, cv2.COLOR_BGR2RGB))
                    num += 1
        else:
            pbar = tqdm(enumerate(dataloader_val), total=len(ds_test)//BATCH_SIZE)
            for idx, hr in pbar:
                # if idx == 5:
                #     break
                # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video

                hr = hr.float().permute(0, 1, 4, 2, 3) / 255.
                hr = hr[:, :, ...]

                # 1 b c h w
                transform1 = T.Resize((hr.shape[3]//4, hr.shape[4]//4))
                hr_trans = transform1(hr.squeeze(0)).unsqueeze(0)
                hr_trans = hr_trans.to(device)

                # y pred
                # 1 b c h w
                start = time.time()
                y_pred = model(hr_trans).squeeze().permute(0, 2, 3, 1).cpu().numpy()
                end = time.time()
                print("The time of execution:",
                    (end-start) * 10**3, "ms")
                #print("Shape: ", lr.shape, y_pred.shape)
                # imgplot = plt.imshow(y_pred[1])
                # plt.show()
                hr_trans = hr_trans.squeeze().permute(0, 2, 3, 1).cpu().numpy()
                hr = hr.squeeze().permute(0, 2, 3, 1).cpu().numpy()
                for i, pred in enumerate(y_pred):
                    cv2.imwrite('/home/moksyasha/Projects/SkyScale/MoksVSR/results/vid_plus_hr/{:06d}.png'.format(num), cv2.cvtColor(255*hr[i], cv2.COLOR_BGR2RGB))
                    cv2.imwrite('/home/moksyasha/Projects/SkyScale/MoksVSR/results/vid_plus_pred/{:06d}.png'.format(num), cv2.cvtColor(255*pred, cv2.COLOR_BGR2RGB))
                    num += 1
            


if __name__ == '__main__':
    main()
