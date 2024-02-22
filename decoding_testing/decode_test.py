import cv2
import time
import nvtx
import torch
import torchvision.transforms as transforms
import torch
import torchvision
import imageio as io
from torch.profiler import profile, record_function, ProfilerActivity

@nvtx.annotate("opencv", color="purple")
def main():
    #video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test2560x1440_24_5sec.mp4' #2k
    video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test1920x1080_24_5sec.mp4' #fhd
    #video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test1280x720_24_5sec.mp4' #hd
    # video_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/own/test480_270.mp4'
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    frame_count = 0
    start_time = time.time()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    device = torch.device("cuda")

    while True:
        
        ret, frame = cap.read()

        if not ret:
            break

        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_tensor = transform(frame).unsqueeze(0)
        
        frame_count += 1
        image_tensor = image_tensor.to(device)

        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break

    end_time = time.time()
    total_time = end_time - start_time

    cap.release()
    cv2.destroyAllWindows()

    print("cv\nTotal frames read:", frame_count)
    print("Total time:", total_time, "seconds")
    print("FPS:", frame_count / total_time)

from torchvision.transforms import v2


@nvtx.annotate("cuda", color="green")
def main_cuda():
    torch.cuda.cudart().cudaProfilerStart()
    #video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test2560x1440_24_5sec.mp4' #2k
    video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test1920x1080_24_5sec.mp4' #fhd
    #video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test1280x720_24_5sec.mp4' #hd

    start_time = time.time()
    frame_count = 0

    transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    torchvision.set_video_backend("cuda")
    reader = torchvision.io.VideoReader(video_path, "video")

    while True:
        try:
            frame = ((next(reader))["data"]).float().permute(2, 0, 1)

            frame = transform(frame)
            frame_count += 1
        except Exception as e:
            print(e)
            break

    end_time = time.time()
    total_time = end_time - start_time
    torch.cuda.cudart().cudaProfilerStop()
    print("cuda\nTotal frames read:", frame_count)
    print("Total time:", total_time, "seconds")
    print("FPS:", frame_count / total_time)


@nvtx.annotate("io", color="yellow")
def main_io():

    #video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test2560x1440_24_5sec.mp4' #2k
    video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test1920x1080_24_5sec.mp4' #fhd
    #video_path = '/home/moksyasha/Projects/SkyScale/decoding_testing/test1280x720_24_5sec.mp4' #hd
    # video_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/own/test480_270.mp4'

    frame_count = 0
    start_time = time.time()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    device = torch.device("cuda")

    for frame in io.imiter(video_path):

        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_tensor = transform(frame).unsqueeze(0)
        
        frame_count += 1
        image_tensor = image_tensor.to(device)

        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(30) & 0xFF == ord('q'):
        #     break

    end_time = time.time()
    total_time = end_time - start_time

    print("io\nTotal frames read:", frame_count)
    print("Total time:", total_time, "seconds")
    print("FPS:", frame_count / total_time)

    

if __name__ == '__main__':
    main()
    main_cuda()
    main_io()
    #ff()
