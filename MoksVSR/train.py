from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os, glob, multiprocessing, torch
#from model.BasicVSR import *
import matplotlib.pyplot as plt
import time
from flow_vis import flow_to_color
import cv2

#sys.path.append(os.path.abspath("/home/moksyasha/Projects/SkyScale/MoksVSR/models/moksvsr"))
from models.moksvsr.MoksVSR import MoksVSR, MoksPlus, TestMoksVSRDouble
from dataset import RedsDataset


def main():
    torch.cuda.empty_cache()
    BATCH_SIZE = 1
    EPOCHS = 30
    N_CORES = multiprocessing.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_path = '/media/moksyasha/linux_data/datasets/REDS4/train_sharp'
    ds = RedsDataset(ds_path, 14)
    dataloader_train = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    ds_path = '/media/moksyasha/linux_data/datasets/REDS4/val_sharp'
    ds_val = RedsDataset(ds_path, 14)
    dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    model = MoksPlus()
    checkpoint = torch.load("/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/plus_resize64_4.pt")
    #model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    opt = Adam(model.parameters(), lr=2e-4)
    charbonnier_loss_fn = lambda y_pred, y_true, eps=1e-8: torch.mean(torch.sqrt((y_pred - y_true)**2 + eps**2)) # essentially equal to torch.abs(y_pred-y_true) since eps is small
    
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    # transform1 = T.CenterCrop((64, 64))
    # transform2 = T.CenterCrop((256, 256))
    transform1 = T.Resize((64, 64))
    transform2 = T.Resize((256, 256))
    #transform1 = T.Resize((128, 256))
    #transform2 = T.Resize((512, 1024))
    epochs = []
    # fig, ax = plt.subplots(2, 3)
    # lr, hr = ds[0]
    # lr_c = torch.from_numpy(lr[0]).permute(2, 0, 1).unsqueeze(0)
    # lr_c1 = transform1(lr_c).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    # lr_c2 = transform3(lr_c).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    # hr_c = torch.from_numpy(hr[0]).permute(2, 0, 1).unsqueeze(0)
    # hr_c1 = transform2(hr_c).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    # hr_c2 = transform4(hr_c).permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
    # ax[0,0].imshow(lr_c.permute(0, 2, 3, 1).squeeze(0).numpy())
    # ax[0,1].imshow(lr_c1)
    # ax[0,2].imshow(lr_c2)
    # ax[1,0].imshow(hr_c.permute(0, 2, 3, 1).squeeze(0).numpy())
    # ax[1,1].imshow(hr_c1)
    # ax[1,2].imshow(hr_c2)
    # plt.show()
    # exit()
    train_model_training_loss_ls = []
    train_model_training_accuracy_ls = []
    validation_model_training_loss_ls = []
    validation_model_training_accuracy_ls = []

    for epoch in range(EPOCHS):
        num = 0
        losses = 0
        pbar = tqdm(enumerate(dataloader_train), total=len(ds)//BATCH_SIZE)
        for idx, (lr, hr) in pbar:
            torch.cuda.empty_cache()
            # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
            #start = time.time()
            #transform = T.Resize((180, 320))

            lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., lr.float().permute(0, 1, 4, 2, 3) / 255.        # normalize frames
            lr = transform1(lr.squeeze(0)).unsqueeze(0)
            hr = transform2(hr.squeeze(0)).unsqueeze(0)
            #lr = (lr - mean) / std
            lr, hr = lr.to(device), hr.to(device)
            
            # pred = model.forward(lr)
            # imgplot = plt.imshow(pred[0][0].permute(1, 2, 0).detach().cpu().numpy())
            # plt.show()
            # print(pred.shape)
            # end = time.time()
            # print("The time of execution:",
            #     (end-start) * 10**3, "ms")
            num += 1
            model.train()
            opt.zero_grad()
            y_pred = model.forward(lr)
            loss = charbonnier_loss_fn(y_pred, hr)
            loss.backward()
            losses += loss.item()
            opt.step()
            pbar.set_description(f'Epoch {epoch}, loss: {round(float(loss), 5)}')
        a = losses/num
        print("=TRAIN LOSS: ", a)
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
                lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., lr.float().permute(0, 1, 4, 2, 3) / 255.        # normalize frames
                lr = transform1(lr.squeeze(0)).unsqueeze(0)
                hr = transform2(hr.squeeze(0)).unsqueeze(0)
                #lr = (lr - mean) / std
                lr, hr = lr.to(device), hr.to(device)
                y_pred = model(lr)
                loss = charbonnier_loss_fn(y_pred, hr)
                losses += loss.item()
                num += 1
        a = losses/num
        print("=VAL LOSS: ", a)
        validation_model_training_loss_ls.append(a)
        
        train_loss = np.array([])
        np.savetxt(f'plus_resize64_valid_{str(epoch)}.txt', np.array(validation_model_training_loss_ls))
        np.savetxt(f'plus_resize64_train_{str(epoch)}.txt', np.array(train_model_training_loss_ls))
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'epoch': epoch
        }, "plus_resize64_" + str(epoch) + ".pt")

    print("train: ", train_model_training_loss_ls)
    print("val: ", validation_model_training_loss_ls)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_model_training_loss_ls, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, validation_model_training_loss_ls, label='Validation Loss', color='red', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plus_resize64_valid.png')


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
