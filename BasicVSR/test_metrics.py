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
import nvtx

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img


def inference(frames, model, save_path, index_img):
    test_results_folder = OrderedDict()
    test_results_folder['psnr'] = []
    test_results_folder['ssim'] = []
    test_results_folder['psnr_y'] = []
    test_results_folder['ssim_y'] = []
    test_results['erqa'] = []
    metric = erqa.ERQA()
    with torch.no_grad():
        start = time.time()
        outputs = model(frames)
        end = time.time() - start
        print("== Time for 5: ", end)
        # save upscaled images to jpg
        for output in outputs:
            output = tensor2img(output, True)

            #save_image(outputs[i], os.path.join(save_path, f"{index_img:04d}_BasicVSR.jpg"))
            #index_img += 1
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

@nvtx.annotate("main", color="purple")
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/moksyasha/Projects/SkyScale/OwnBasicVSR/trained_models/BasicVSR_Vimeo90K.pth')
    parser.add_argument(
        '--input_path', type=str, default='/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/own/test480_270.mp4', help='input test video')
    parser.add_argument('--save_path', type=str, default='/home/moksyasha/Projects/SkyScale/results/BasicVSR/own/temp/', help='save image path')
    parser.add_argument('--interval', type=int, default=100, help='interval size')
    args = parser.parse_args()

    device_cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # set up model
    model = BasicVSR(num_feat=64, num_block=30)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device_cuda)

    os.makedirs(args.save_path, exist_ok=True)
    
    torchvision.set_video_backend("cuda")
    reader = torchvision.io.VideoReader(args.input_path, "video")
    fps = reader.get_metadata()['video']['fps']

    test_frame = (next(reader))["data"]#.permute(0,1,2).cpu().numpy() # 3 1080 1920 // 15 such frames = 100 mb

    frames_per_cycle = 10
    index_img = 0
    start = time.time()

    while True:
        torch.cuda.empty_cache()
        try:
            frames = 0
            curr_frames = torch.empty((frames_per_cycle, test_frame.shape[0], test_frame.shape[1], test_frame.shape[2]), dtype=torch.float32, device=device_cuda)
            for i in range(frames_per_cycle):
                curr_frames[i] = (next(reader))["data"] / 255.
                frames += 1
        except:
            if frames:
                curr_frames = curr_frames[:frames].unsqueeze(0).permute(0, 1, 4, 2, 3)
                index_img = inference(curr_frames, model, args.save_path, index_img)
            break
        else:
            curr_frames = curr_frames.unsqueeze(0).permute(0, 1, 4, 2, 3)
            index_img = inference(curr_frames, model, args.save_path, index_img)
        print(f"{index_img} frames upscaled")

    end = time.time() - start
    print("== Time for single video: ", end)
    os.chdir(args.save_path)
    os.system('ffmpeg -r 24 -pattern_type glob -i "*.jpg" -c:v libx264 -movflags +faststart ../output.mp4')
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
