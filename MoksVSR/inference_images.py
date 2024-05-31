import os, glob, multiprocessing, torch
from torch.utils.data import DataLoader
from models.moksvsr.MoksVSR import MoksVSR, MoksPlus
from dataset import VidDataset, RedsDataset
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

    ds_test = RedsDataset('/media/moksyasha/linux_data/datasets/REDS4/test_sharp', 2)
    #dataloader_val = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CORES)
    #ds_test = VidDataset('/media/moksyasha/linux_data/datasets/Vid4', 20)
    #dataloader_val = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_CORES)

    # own model
    model = MoksPlus()
    #model_2 = Generator()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    model = model.to(device)

    model.eval()  # handle drop-out/batch norm layers
    # torch.cuda.empty_cache()
    # lr, hr = ds_test[0]
    # lr_gpu, hr_gpu = torch.from_numpy(lr).float() / 255., torch.from_numpy(hr).float() / 255.
    # lr_gpu = lr_gpu.permute(0, 3, 1, 2)
    # transform1 = T.Resize((128, 128))
    # #lr_gpu = transform1(lr_gpu)
    # lr_gpu = lr_gpu.to(device)
    #y_pred = model(lr_gpu.unsqueeze(0).permute(0, 1, 4, 2, 3))
    
    ## test 2 images
    img1_path = "/media/moksyasha/linux_data/01.png"
    img2_path = "/media/moksyasha/linux_data/02.png"
    img1 = torch.tensor(cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)).float().unsqueeze(0)/255.0
    img2 = torch.tensor(cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)).float().unsqueeze(0)/255.0
    transform2 = T.Resize((img1.shape[1]//4, img1.shape[2]//4))
    imgss = torch.cat((img1, img2), dim=0).permute(0, 3, 1, 2)
    imgss = transform2(imgss)
    imgss = imgss.to(device)
    ##
    start = time.time()
    y_pred1 = model(imgss.unsqueeze(0)).squeeze().permute(0, 2, 3, 1).detach().cpu().numpy()
    print(y_pred1.shape)
    end = time.time()
    print("The time of execution:",
        (end-start) * 10**3, "ms")
    fig, ax = plt.subplots(1, 3)
    a_cpu = imgss[0].permute(1, 2, 0).detach().cpu().numpy()

    ax[0].imshow(a_cpu)
    ax[1].imshow(y_pred1[0])
    ax[2].imshow(img2.squeeze())
    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.show()
            # for pred in y_pred:
            #     cv2.imwrite('/home/moksyasha/Projects/SkyScale/MoksVSR/results/1/{:06d}.png'.format(num), cv2.cvtColor(255*pred, cv2.COLOR_BGR2RGB))
            #     num += 1
            


if __name__ == '__main__':
    main()