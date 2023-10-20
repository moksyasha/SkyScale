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
 
def preprocess_video_gpu(video_path, image_size_low, frames_per_cycle=30):
   reader = torchvision.io.VideoReader(video_path, "video", num_threads=0, device="cuda")
   resizer = torchvision.transforms.Resize(image_size_low, antialias=True)
 
   curr_frames = []
   all_frames_low = []
 
   for frame in reader:
       curr_frames.append(frame["data"])
 
       if len(curr_frames) == frames_per_cycle:
           resize_chunk(curr_frames, all_frames_low, resizer)
           curr_frames = []
 
   if len(curr_frames) > 0:
       resize_chunk(curr_frames, all_frames_low, resizer)
 
   all_frames_low = torch.cat(all_frames_low, 0)
 
   return all_frames_low


def resize_chunk(curr_frames, all_frames_low, resizer):
   curr_frames = torch.stack(curr_frames, 0)
   curr_frames = curr_frames.permute(0, 3, 1, 2)
   curr_frames_low = resizer(curr_frames)
   curr_frames_low = curr_frames_low.permute(0, 2, 3, 1)
   all_frames_low.append(curr_frames_low)

# from torchvision.io import _HAS_VIDEO_OPT
# print(_HAS_VIDEO_OPT)
import distutils.spawn
import shutil
print(shutil.which('ffmpeg'))
print(distutils.spawn.find_executable('ffmpeg'))
# s = time.time()

# #torchvision.set_video_backend("video_reader")

# stream = "video"
# video_path = "datasets/own/test.mp4"
# video = torchvision.io.VideoReader(video_path, stream)
# video.get_metadata()

# #print("Frames:", frames)
# print("Preprocess on GPU time:", time.time() - s)