import argparse
import cv2
import glob
import os
import shutil
import torch
import time
import nvtx

from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img

@nvtx.annotate("main", color="green")
def inference(imgs, imgnames, model, save_path):
    start = time.time()
    with torch.no_grad():
        outputs = model(imgs)
    end = time.time()
    print(end - start)
    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output, imgname in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{imgname}_BasicVSR.png'), output)


@nvtx.annotate("main", color="purple")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/moksyasha/Projects/SkyScale/OwnBasicVSR/trained_models/BasicVSR_Vimeo90K.pth')
    parser.add_argument(
        '--input_path', type=str, default='/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/own/photo', help='input test image folder')
    parser.add_argument('--save_path', type=str, default='/home/moksyasha/Projects/SkyScale/results/BasicVSR/orig/temp/', help='save image path')
    parser.add_argument('--interval', type=int, default=15, help='interval size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.model_path)
    # set up model
    model = BasicVSR(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # extract images from video format files
    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

    start_main = time.time()
    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    print(num_imgs)
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, args.save_path)
    else:
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True) #imgs 15 3 180 320
            imgs = imgs.unsqueeze(0).to(device) #imgs 1 15 3 180 320
            inference(imgs, imgnames, model, args.save_path)

    end = time.time()
    print("Total secs:  ", end - start_main)
    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    main()