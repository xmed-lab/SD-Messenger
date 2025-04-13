from dataset.acdc_transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box
import matplotlib.pyplot as plt
from matplotlib import cm
from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange, repeat


class ACDCDataset(Dataset):

    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/valtest.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]

        # print(f'img shape: {img.shape}')
        # print(f'mask shape: {mask.shape}')
        # print(f'seg class of {id}: {np.unique(mask)}')

        # print(f'img rgb: {img_rgb.shape}')
        if self.mode == 'val':
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), id

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        # print(f'img rgb: {img.shape}')
        if self.mode == 'train_l':
            img = torch.from_numpy(img).unsqueeze(0).float()
            img = repeat(img, 'c h w -> (repeat c) h w', repeat=3)
            return img, torch.from_numpy(np.array(mask)).long()

        # img = Image.fromarray((img * 255).astype(np.uint8))
        # img_path = os.path.join(self.root, id).replace('data/slices/', 'img/').replace('h5', 'png')
        # img.save(f'{img_path}')
        # mask = Image.fromarray((mask).astype(np.uint8))
        # mask_path = os.path.join(self.root, id).replace('data/slices/', 'mask/').replace('h5', 'png')
        # mask.save(f'{mask_path}')

        # img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float()
        img = repeat(img, 'c h w -> (repeat c) h w', repeat=3)
        # print(img.shape)
        # if random.random() < 0.8:
        #     img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        # img_s1 = blur(img_s1, p=0.5)
        # cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        # img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0
        #
        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        # img_s2 = blur(img_s2, p=0.5)
        # cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        # img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        return img

    def __len__(self):
        return len(self.ids)
