#!/usr/bin/python

# -*- coding: utf8 -*-

import torch
from utils_gan import *
from PIL import Image, ImageDraw, ImageFont
import time
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model checkpoints
srgan_checkpoint = "GAN/checkpoint_srgan.pth.tar"

# Load GAN model
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()
#320 180
#1280 720
#2 073 600 = 1920 * 1080 fullhd
#3 686 400 = 2560 * 1440 2k
#8 294 400 = 3840 * 2160 4k

def visualize_sr(img):
    """
    Visualizes the super-resolved images from the SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR iamge
    """
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)
    
    print("Size lr: ", lr_img.size)
    print("Size hr: ", hr_img.size)
    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolution (SR) with SRGAN
    start = time.time()
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    end = time.time() - start
    print("== Time for single image: ", end)
    # Create grid
    margin = 40
    grid_img = Image.new('RGB', (2 * hr_img.width + 3 * margin, 2 * hr_img.height + 3 * margin), (255, 255, 255))

    # Font
    draw = ImageDraw.Draw(grid_img)
    try:
        font = ImageFont.truetype("calibril.ttf", size=23)
        # It will also look for this file in your OS's default fonts directory, where you may have the Calibri Light font installed if you have MS Office
        # Otherwise, use any TTF font of your choice
    except OSError:
        print(
            "Defaulting to a terrible font. To use a font of your choice, include the link to its TTF file in the function.")
        font = ImageFont.load_default()

    # Place bicubic-upsampled image
    grid_img.paste(bicubic_img, (margin, margin))
    text_size = font.getsize("Bicubic")
    draw.text(xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, margin - text_size[1] - 5], text="Bicubic",
              font=font,
              fill='black')

    # Place SRGAN image
    grid_img.paste(sr_img_srgan, (2 * margin + bicubic_img.width, margin))
    text_size = font.getsize("SRGAN")
    draw.text(
        xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, 2 * margin + bicubic_img.height - text_size[1] - 5],
        text="SRGAN", font=font, fill='black')

    # Place original HR image
    grid_img.paste(hr_img, (2 * margin + bicubic_img.width, 2 * margin + bicubic_img.height))
    text_size = font.getsize("Original HR")
    draw.text(xy=[2 * margin + bicubic_img.width + bicubic_img.width / 2 - text_size[0] / 2,
                  2 * margin + bicubic_img.height - text_size[1] - 1], text="Original HR", font=font, fill='black')

    # Display grid
    grid_img.show()

    return grid_img


if __name__ == '__main__':
    torch.cuda.empty_cache()
    grid_img = visualize_sr("GAN/media/4.jpg")