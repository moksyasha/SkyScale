import argparse
import cv2
import glob
import sys
import os
import shutil
import torch
import torchvision
from torchvision.utils import save_image
import time
import subprocess
import numpy as np

from models.moksvsr.MoksVSR import MoksVSR, MoksPlus

def inference(frames, model, save_path, index_img):
    with torch.no_grad():
        start = time.time()
        outputs = model(frames)
        end = time.time() - start
        print("== Time for 10: ", end)
        # save upscaled images to jpg
        outputs = outputs.squeeze()
        for i in range(outputs.shape[0]):
            save_image(outputs[i], os.path.join(save_path + "temp/", f"{index_img:04d}_MoksPlus.jpg"))
            index_img += 1
            #torchvision.io.write_jpeg(outputs[i], os.path.join(save_path, f'{index_img}_BasicVSR.jpg'), 100)
    # save imgs
    # outputs = outputs.squeeze()
    # outputs = list(outputs)
    # for output in outputs:
    #     output = tensor2img(output, True)
        
    #     cv2.imwrite(os.path.join(save_path, f'{index_img}_BasicVSR.jpg'), output)
    #     index_img = index_img + 1
    return index_img

#320 180
#1280 720 hd
#2.073.600 = 1920 * 1080 fullhd
#3.686.400 = 2560 * 1440 2k
#8.294.400 = 3840 * 2160 4k (hd*3) (fhd*2) (2k*1.5)
#                           (hd*9) (fhd*4) (2k*2.25)
#/usr/local/cuda-12.3/nsight-systems-2023.3.3/bin/nsys profile -t nvtx,osrt --force-overwrite=true --stats=true --output=quickstart python ../inference_one_video.py

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/moksyasha/Projects/SkyScale/MoksVSR/checkpoints/moks_resize64_29.pt')
    parser.add_argument(
        '--input_path', type=str, default='/media/moksyasha/linux_data/datasets/own/test270x480_24_1sec.mp4', help='input test video')
    parser.add_argument('--save_path', type=str, default='/home/moksyasha/Projects/SkyScale/MoksVSR/results/video/', help='save image path')
    parser.add_argument('--interval', type=int, default=10, help='interval size')
    args = parser.parse_args()

    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set up model
    model = MoksVSR()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device_cuda)

    os.makedirs(args.save_path + "temp/", exist_ok=True)
    
    torchvision.set_video_backend("cuda")
    reader = torchvision.io.VideoReader(args.input_path, "video")
    dur = reader.get_metadata()['video']['duration']
    fps = reader.get_metadata()['video']['fps']
    frames_all = np.ceil(dur * fps)
    print("all cadr: ", dur * fps)

    test_frame = (next(reader))["data"] #.permute(0,1,2).cpu().numpy() # 3 1080 1920 // 15 such frames = 100 mb
    
    frames_per_cycle = 10
    index_img = 0
    start = time.time()

    curr_frames = torch.empty((frames_per_cycle, test_frame.shape[0], test_frame.shape[1], test_frame.shape[2]), dtype=torch.float32, device=device_cuda)
    while True:
        torch.cuda.empty_cache()
        try:
            frames = 0
            for i in range(frames_per_cycle):
                curr_frames[i] = (next(reader))["data"] / 255.
                frames += 1
        except Exception as e:
            print(e)
            if frames:
                curr_frames = curr_frames[:frames].unsqueeze(0).permute(0, 1, 4, 2, 3)
                index_img = inference(curr_frames, model, args.save_path, index_img)
            break
        else:
            #curr_frames = curr_frames.unsqueeze(0).permute(0, 1, 4, 2, 3)
            index_img = inference(curr_frames.unsqueeze(0).permute(0, 1, 4, 2, 3), model, args.save_path, index_img)
            print(f"{index_img}/{frames_all} frames upscaled")

    end = time.time() - start
    print("== Time for single video: ", end)
    os.chdir(args.save_path + "temp/")
    output_name = "first1"
    os.system(f'ffmpeg -r 24 -pattern_type glob -i "*.jpg" -c:v libx264 -movflags +faststart ../{output_name}.mp4')
    os.system('cd .. && rm -rf ./temp')
    # cmd2 = ['ffmpeg', '-r', '24', '-pattern_type', 'glob', '-i', '"*.jpg"', '-c:v', 'libx264', '-movflags', '+faststart', '../output.mp4']
    # retcode = subprocess.call(cmd2)
    # if not retcode == 0: #or not retcode2 == 0:
    #     raise ValueError('Error {} executing command: {}'.format(retcode, ' '.join(cmd2)))
    # else:
    print("Upscale completed")  



if __name__ == '__main__':
    main()
    #ff()
