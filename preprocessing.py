import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import torch
import numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
from utils import data_augmentation, get_one_hot


class Data(Dataset):
    def __init__(self, data_path, train):
        if train:
            self.data_path = data_path + '/train'
        else:
            self.data_path = data_path + '/valid'
        self.image_dirs = sorted(glob.glob(self.data_path+'/img/*'))
        self.seg_dirs = sorted(glob.glob(self.data_path+'/seg/*'))

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        clean_img = io.imread(image_dir)
        seg_dir = self.seg_dirs[idx]
        seg_map = io.imread(seg_dir)

        clean, seg = ToTensor(clean_img, seg_map)
        clean, seg = RandomCrop(clean, seg)
        clean, seg = Augmentation(clean, seg)
        clean, noisy = AWGN(clean, seg)
        return clean, noisy

def ToTensor(clean_img, seg_map):
    clean = clean_img.transpose((2,0,1))
    clean = torch.tensor(clean).float()/255.0
    seg = torch.tensor(seg_map)
    seg = torch.unsqueeze(seg, 0)
    return clean, seg

def RandomCrop(clean_img, seg_map, patch_size=90):
    h, w = clean_img.shape[1:]

    top = np.random.randint(0, h-patch_size)
    left = np.random.randint(0, w-patch_size)

    clean = clean_img[:, top:top+patch_size, left:left+patch_size]
    seg = seg_map[:, top:top+patch_size, left:left+patch_size]
    return clean, seg

def Augmentation(clean_img, seg_map):
    mode = np.random.randint(0, 7)
    clean = data_augmentation(clean_img, mode)
    seg = data_augmentation(seg_map, mode)
    return clean, seg

def AWGN(clean_img, seg_map, noise=[0,75]):
    clean = clean_img
    stdn = np.random.uniform(noise[0], noise[1])/255.0
    noise_map = torch.ones(1, clean.shape[1], clean.shape[2]) * stdn
    noise_map = noise_map.float()
    add_noise = torch.FloatTensor(clean.size()).normal_(mean=0, std=stdn)
    noisy_img = clean + add_noise

    # seg_map to one hot matrices
    onehot_segmap = get_one_hot(seg_map, 9)

    noisy_img = torch.cat((noisy_img, noise_map, onehot_segmap), 0)
    return clean, noisy_img

def TestAWGN(clean_img, seg_map, sigma):
    clean = clean_img
    sigma = sigma/255.0
    noise_map = torch.ones(1, clean.shape[1], clean.shape[2]) * sigma
    noise_map = noise_map.float()
    add_noise = torch.FloatTensor(clean.size()).normal_(mean=0, std=sigma)
    noisy_img = clean + add_noise

    # Save noisy image
    save_img = torch.clamp(noisy_img, 0., 1.)
    save_img = save_img * 255
    save_img = torch.round(save_img)
    save_img = save_img.permute(1,2,0)

    # seg_map to one hot matrices
    onehot_segmap = get_one_hot(seg_map, 9)

    noisy_img = torch.cat((noisy_img, noise_map, onehot_segmap), 0)
    return clean, noisy_img, save_img


class ValidData(Dataset):
    def __init__(self, data_path, sigma):
        self.data_path = data_path + '/valid'
        self.image_dirs = sorted(glob.glob(self.data_path+'/img/*'))
        self.seg_dirs = sorted(glob.glob(self.data_path+'/seg/*'))
        self.sigma = sigma

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        clean_img = io.imread(image_dir)
        seg_dir = self.seg_dirs[idx]
        seg_map = io.imread(seg_dir)

        clean, seg = ToTensor(clean_img, seg_map)
        clean, noisy, save = TestAWGN(clean, seg, sigma=self.sigma)

        return clean, noisy


class TestData(Dataset):
    def __init__(self, data_path, sigma):
        self.data_path = data_path + '/test'
        self.image_dirs = sorted(glob.glob(self.data_path+'/img/*'))
        self.seg_dirs = sorted(glob.glob(self.data_path+'/seg/*'))
        self.sigma = sigma

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        image_dir = self.image_dirs[idx]
        image_name = image_dir[14:-4]
        clean_img = io.imread(image_dir)
        seg_dir = self.seg_dirs[idx]
        seg_map = io.imread(seg_dir)

        clean, seg = ToTensor(clean_img, seg_map)
        clean, noisy, save = TestAWGN(clean, seg, sigma=self.sigma)

        save_dir = os.path.join('data/test_noisy', str(self.sigma))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        io.imsave(save_dir+'/%s.png' % image_name, save)

        return clean, noisy, image_name
