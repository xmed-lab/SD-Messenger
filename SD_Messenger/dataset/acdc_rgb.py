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

        # print(f'img rgb: {img_rgb.shape}')
        if self.mode == 'val':
            # print(img.shape)
            z, h, w = img.shape
            img_v = []
            for i in range(0, z):
                slice_rgb = self.convert_to_realRGB(img_gray=img[i, :, :])
                slice_rgb = torch.from_numpy(slice_rgb)
                slice_rgb = rearrange(slice_rgb, 'h w c -> c h w')
                img_v.append(slice_rgb)

                # mask_save = Image.fromarray(mask[i, :, :] * 255)
                # mask_path = os.path.join(self.root, id).replace('data/', 'mask/').replace('.h5', f'_slice_{i}.png')
                # # print(mask_path)
                # mask_save.save(mask_path)

            img_tensor = torch.stack(img_v, dim=0).float() / 255.0
            # print(f'stacked feature: {img_tensor.shape}')
            return img_tensor, torch.from_numpy(mask).long()

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        img = self.convert_to_realRGB(img_gray=img)
        # print(f'img rgb: {img.shape}')
        if self.mode == 'train_l':
            img_l = torch.from_numpy(img).float() / 255.0
            img_l = rearrange(img_l, 'h w c -> c h w')
            return img_l, torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img).astype(np.uint8))
        # img_rgb_path = os.path.join(self.root, id).replace('data/slices/', 'rgb/').replace('h5', 'png')
        # img.save(f'{img_rgb_path}')
        # img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = rearrange(img, 'h w c -> c h w')
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
