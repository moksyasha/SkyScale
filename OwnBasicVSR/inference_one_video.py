import argparse
import cv2
import glob
import sys
import os
import shutil
import torch
import torchvision

from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img

index = 0

def inference(frames, model, save_path):
    with torch.no_grad():
        outputs = model(frames)
    # save imgs
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for output in outputs:
        output = tensor2img(output)
        cv2.imwrite(os.path.join(save_path, f'{index}_BasicVSR.png'), output)
        index = index + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='OwnBasicVSR/trained_models/BasicVSR_Vimeo90K.pth')
    parser.add_argument(
        '--input_path', type=str, default='/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/own/test1920x1080_av.mp4', help='input test video')
    parser.add_argument('--save_path', type=str, default='results/BasicVSR/own/', help='save image path')
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

    test_frame = (next(reader))["data"] #.permute(2,0,1) # 3 1080 1920 // 15 such frames = 100 mb
    print(test_frame.shape[0])
    frames_per_cycle = 5
    curr_frames = torch.empty((frames_per_cycle, test_frame.shape[0], test_frame.shape[1], test_frame.shape[2]), dtype=torch.float32, device=device_cuda)
    for i in range(frames_per_cycle):
        curr_frames[i] = (next(reader))["data"]
    curr_frames = curr_frames.unsqueeze(0).permute(0, 1, 4, 2, 3)
    pass
    #print(curr_frames.unsqueeze(0).permute(0, 1, 4, 2, 3).shape) # torch.Size([1, 15, 3, 1080, 1920])
    inference(curr_frames, model, args.save_path)
    


if __name__ == '__main__':
    main()
