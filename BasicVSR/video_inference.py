import cv2
import time
import numpy as np


# Output: 692.3 frames per second

cap.release()
cv2.destroyAllWindows()

import argparse
import cv2
import glob
import os
import shutil
import torch

from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def inference(imgs, imgnames, model, save_path):
    import time

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='OwnBasicVSR/trained_models/BasicVSR_Vimeo90K.pth')
    parser.add_argument(
        '--input_path', type=str, default='OwnBasicVSR/datasets/own/test.mp4', help='input test video')
    parser.add_argument('--save_path', type=str, default='results_video/BasicVSR', help='save image path')
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


    cap = cv2.VideoCapture("OwnBasicVSR/datasets/own/test.mp4", apiPreference=cv2.CAP_FFMPEG)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
    #else:
        #cv2.imshow('Frame', frame)

        #if cv2.waitKey(100) & 0xFF == ord('q'):
        #    break
    #assert frame.shape == (720, 1280, 3)
    #assert frame.dtype == np.uint8
    end = time.perf_counter()

    print(f"Frames: {frames}")
    print(f"{frames/(end-start):.1f} frames per second")


    # extract images from video format files
    input_path = args.input_path
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(args.input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {args.input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

    # load data and inference
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    if len(imgs_list) <= args.interval:  # too many images may cause CUDA out of memory
        imgs, imgnames = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, args.save_path)
    else:
        for idx in range(0, num_imgs, args.interval):
            interval = min(args.interval, num_imgs - idx)
            imgs, imgnames = read_img_seq(imgs_list[idx:idx + interval], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, args.save_path)

    # delete ffmpeg output images
    if use_ffmpeg:
        shutil.rmtree(input_path)


if __name__ == '__main__':
    main()
