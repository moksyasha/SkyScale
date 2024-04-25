import os, glob, multiprocessing, torch
import cv2
from PIL import Image
from dataset import VidDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
from utils import utils_image as util
import math
import torchvision.transforms as T
from skimage.metrics import structural_similarity as compare_ssim


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def main():

    path_hr = "/home/moksyasha/Projects/SkyScale/MoksVSR/results/vid_plus_hr/"
    path_pred = "/home/moksyasha/Projects/SkyScale/MoksVSR/results/vid_plus_pred/"
    path_hr = sorted(glob.glob(f'{path_hr}/*'))
    path_pred = sorted(glob.glob(f'{path_pred}/*'))
    # img = np.array(Image.open(images_path[0]))
    # lr_y, lr_x, chan = img.shape
    # pred = np.zeros((len(images_path), lr_y, lr_x, chan), dtype=np.uint8)
    # for i, path in enumerate(images_path):
    #     img = np.array(Image.open(images_path[i]))
    #     pred[i] = img

    test_results_folder = {}
    test_results_folder['psnr'] = []
    test_results_folder['ssim'] = []
    test_results_folder['erqa'] = []
    test_results_folder['lpips'] = []

    num = 0

    # import erqa
    # metric_erqa = erqa.ERQA()
    # import lpips
    # loss_fn_alex = lpips.LPIPS(net='alex')

    for i in range(len(path_hr)):

        gt_np = np.array(Image.open(path_hr[i]))
        pred_np = np.array(Image.open(path_pred[i]))
        # 720 1280 3 pred
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(pred[0])
        # ax[1].imshow(gt)
        # plt.show()
        
        # if num == len(images_path)//10:
        #     break

        #img = bgr2ycbcr(pred[i].astype(np.float32) / 255.) * 255.
        #gt = bgr2ycbcr(gt.cpu().numpy().astype(np.float32) / 255.) * 255.

        #pred_tensor = torch.from_numpy(pred[num]).permute(2, 0, 1).float().unsqueeze(0)
        #gt_lpips = gt.permute(2, 0, 1).unsqueeze(0).float()
        # gt_lpips -= gt_lpips.min(1, keepdim=True)[0]
        # gt_lpips /= gt_lpips.max(1, keepdim=True)[0]

        # pred_tensor -= pred_tensor.min(1, keepdim=True)[0]
        # pred_tensor /= pred_tensor.max(1, keepdim=True)[0]
        # gt_lpips /= 255.
        # pred_tensor /= 255.

        #transform = T.Compose([
        #    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        #pred_tensor = transform(pred_tensor)
        #gt_lpips = transform(gt_lpips)
        pnsr = cv2.PSNR(pred_np, gt_np)
        if pnsr < 50:
            test_results_folder['psnr'].append(cv2.PSNR(pred_np, gt_np))
        test_results_folder['ssim'].append(calculate_ssim(pred_np, gt_np))

        num += 1
    
    psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
    ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
    # erqa = sum(test_results_folder['erqa']) / len(test_results_folder['erqa'])
    # lpips = sum(test_results_folder['lpips']) / len(test_results_folder['lpips'])
    print(psnr, ssim)
    exit()
    # lr_out = np.zeros((, lr_y, lr_x, 3), dtype=np.uint8)
    # lr_img = np.array(Image.open(lr_path))
    # imgplot = plt.imshow(y_pred[1])
    # plt.show()

if __name__ == '__main__':
    main()
