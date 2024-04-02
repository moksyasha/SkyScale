from torch.utils.data import Dataset, DataLoader
import os
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and eval) a model')
    parser.add_argument('--config', default="/home/moksyasha/Projects/SkyScale/MoksVSR/config/configs/basicvsr/basicvsr_2xb4_reds4.py", help='test config file path')
    parser.add_argument('--checkpoint', default="/home/moksyasha/Projects/SkyScale/MoksVSR/models_trained/basicvsr_reds4.pth", help='checkpoint file')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    # parser.add_argument(
    #     '--cfg-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='override some settings in the used config, the key-value pair '
    #     'in xxx=yyy format will be merged into config file. If the value to '
    #     'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    #     'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    #     'Note that the quotation marks are necessary and that no white space '
    #     'is allowed.')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    # # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # # will pass the `--local-rank` parameter to `tools/train.py` instead
    # # of `--local_rank`.
    # parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class VSRDataset(Dataset):
    def __init__(self, path, imgs_per_clip=20):
        '''
        Args:
        lr_path (str): Represents a path that contains a set of folders,
            where each folder contains a sequence of
            consecutive LR frames of a video.
        hr_path (str): See lr_path, but each folder
            contains HR frames. The folder and image names,
            when sorted, should be a 1:1 match with the LR frames
            (i.e. the third image in the second folder of the lr_path
            should be the LR image ofthe third image
            in the second folder of the hr_path).
        imgs_per_clip (int): The number of images that
            represents an input video. Default is 15,
            meaning each input video will consist of
            15 consecutive frames.
        '''
        self.dataset_path = f'{path}'
        self.imgs_per_clip = imgs_per_clip
        self.lr_folders = sorted(glob.glob(f'{self.lr_path}/*') )
        self.hr_folders = sorted(glob.glob(f'{self.hr_path}/*'))
        self.clips_per_folder = len(glob.glob(f'{self.lr_folders[0]}/*')) // imgs_per_clip
        self.num_clips = len(self.lr_folders) * self.clips_per_folder

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        '''
        Returns a np.array of an input video that is of shape
        (T, H, W, 3), where T = imgs per clip, H/W = height/width,
        and 3 = channels (3 for RGB images). Note that the video
        pixel values will be between [0, 255], not [0, 1).
        '''
        folder_idx, clip_idx = idx // self.clips_per_folder, idx % self.clips_per_folder
        s_i, e_i = self.imgs_per_clip * clip_idx, self.imgs_per_clip * (clip_idx + 1)
        lr_fnames = sorted(glob.glob(f'{self.lr_folders[folder_idx]}/*'))[s_i:e_i]
        hr_fnames = sorted(glob.glob(f'{self.hr_folders[folder_idx]}/*'))[s_i:e_i]
        
        for i, (lr_fname, hr_fname) in enumerate(zip(lr_fnames, hr_fnames)):
            lr_img = np.array(Image.open(lr_fname))
            hr_img = np.array(Image.open(hr_fname))
            if i == 0: # instantiate return LR and HR arrays if loading the first image of the batch
                lr_res_y, lr_res_x, _c = lr_img.shape
                hr_res_y, hr_res_x, _c = hr_img.shape
                lr = np.zeros((self.imgs_per_clip, lr_res_y, lr_res_x, 3), dtype=np.uint8)
                hr = np.zeros((self.imgs_per_clip, hr_res_y, hr_res_x, 3), dtype=np.uint8)
            lr[i], hr[i] = lr_img, hr_img
        
        return lr, hr


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/REDS4/test_sharp_bicubic'
    self.folders = sorted(glob.glob(f'{self.ds_path}/*'))
    #ds = VSRDataset(ds_path)
    #dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    # ds_path = '/home/moksyasha/Projects/SkyScale/OwnBasicVSR/datasets/REDS4/val_sharp_bicubic'
    # ds_val = VSRDataset(ds_path)
    # dataloader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES)

    # model = Generator().to(device)
    # opt = Adam(model.parameters(), lr=2e-4)
    # charbonnier_loss_fn = lambda y_pred, y_true, eps=1e-8: torch.mean(torch.sqrt((y_pred - y_true)**2 + eps**2)) # essentially equal to torch.abs(y_pred-y_true) since eps is small

    # train_model_training_loss_ls = []
    # train_model_training_accuracy_ls = []
    # validation_model_training_loss_ls = []
    # validation_model_training_accuracy_ls = []
    # epochs = []
    
    # losses = 0
    
    # for epoch in range(EPOCHS):
    #     num = 0
    #     losses = 0
    #     pbar = tqdm(enumerate(dataloader), total=len(ds)//BATCH_SIZE)
    #     for idx, (lr, hr) in pbar:
    #         # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
    #         lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
    #         lr, hr = lr.to(device), hr.to(device)
    #         num += 1
    #         model.train()
    #         opt.zero_grad()
    #         y_pred = model(lr)
    #         loss = charbonnier_loss_fn(y_pred, hr)
    #         loss.backward()
    #         losses += loss.item()
    #         opt.step()
    #         pbar.set_description(f'Epoch {epoch}, loss: {round(float(loss), 5)}')
    #     a = losses/num
    #     print(a)
    #     epochs.append(epoch)
    #     train_model_training_loss_ls.append(a)
    #     num = 0

    #     # valid
    #     model.eval()  # handle drop-out/batch norm layers
    #     losses = 0
    #     num = 0
    #     with torch.no_grad():
    #         pbar = tqdm(enumerate(dataloader_val), total=len(ds_val)//BATCH_SIZE)
    #         for idx, (lr, hr) in pbar:
    #             # NTHWC -> NTCHW, where T = time dimension = number of frames per training input video
    #             lr, hr = lr.float().permute(0, 1, 4, 2, 3) / 255., hr.float().permute(0, 1, 4, 2, 3) / 255.
    #             lr, hr = lr.to(device), hr.to(device)
    #             y_pred = model(lr)
    #             loss = charbonnier_loss_fn(y_pred, hr)
    #             losses += loss.item()
    #             num += 1
    #     a = losses/num
    #     print(a)
    #     validation_model_training_loss_ls.append(a)

    #     if epoch % 5 == 0:
    #         torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': opt.state_dict(),
    #         'epoch': epoch
    #         }, "val_loss_basic_chkp_" + str(epoch) + ".pt")
    
    # print("train: ", train_model_training_loss_ls)
    # print("val: ", validation_model_training_loss_ls)

    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, train_model_training_loss_ls, label='Training Loss', color='blue', marker='o')
    # plt.plot(epochs, validation_model_training_loss_ls, label='Validation Loss', color='red', marker='o')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
