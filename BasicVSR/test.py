# import os
# import ffmpeg

# input_path_orig = "OwnBasicVSR/datasets/own/test.mp4"
# video_name = os.path.splitext(os.path.split(input_path_orig)[-1])[0]
# input_path = os.path.join('./BasicVSR_tmp', video_name)
# os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
# ffmpeg.input(input_path_orig, ss = 0, r = 1)\
#         .filter('fps', fps='1/60')\
#         .output('thumbs/test-%d.jpg', start_number=0)\
#         .overwrite_output()\
#         .run(quiet=True)
# #os.system(f'ffmpeg -i {input_path_orig} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  {input_path} /frame%08d.png')

import torch
import torchvision
import time
import numpy as np
import cv2

from torchvision.io.video_reader import _HAS_VIDEO_OPT
from torchvision.io._load_gpu_decoder import _HAS_GPU_VIDEO_DECODER
print(_HAS_VIDEO_OPT)
print(_HAS_GPU_VIDEO_DECODER)
print(torchvision.io._load_gpu_decoder.__file__)
# s = time.time()
# frames = 0
# torchvision.set_video_backend("cuda")
# video_path = "/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/own/test1920x1080_av.mp4"
# reader = torchvision.io.VideoReader(video_path, "video")

# for frame in reader:
#     frames+=1

# print("torchFrames:", frames)
# print("Preprocess on GPU time:", time.time() - s)
# end = time.time()
# print(f"{frames/(end-s):.1f} frames per second")

# frames = 0
# # 123?
# cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
# frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# start = time.time()
# while True:
#     ret, frame = cap.read()
    
#     if ret is False:
#         break
#     else:
#         img = torch.from_numpy(frame).float().to(device)

# end = time.time()
# print("Preprocess on GPU time:", end - start)
# print(f"{frames/(end-start):.1f} frames per second")

# cap.release()