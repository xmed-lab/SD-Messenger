from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange, repeat

class SemiDataset(Dataset):
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
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        # img for cutmix
        mix_item = random.choice(range(self.__len__()))
        mix_id = self.ids[mix_item]
        mix_img = Image.open(os.path.join(self.root, mix_id.split(' ')[0])).convert('RGB')
        mix_mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, mix_id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        mix_img, mix_mask = resize(mix_img, mix_mask, (0.5, 2.0))
        # ignore_value = 254 if self.mode == 'train_u' else 255

        # add 254 into padding even it is for labeled data
        ignore_value = 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)
        mix_img, mix_mask = crop(mix_img, mix_mask, self.size, ignore_value)
        mix_img, mix_mask = hflip(mix_img, mix_mask, p=0.5)

        if self.mode == 'train_u':
            return normalize(img)

        img_s1 = deepcopy(img)
        mix_img_s1 = deepcopy(mix_img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            mix_img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(mix_img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box = obtain_cutmix_box(img_s1.size[0], p=0.5)

        mix_img_s1 = transforms.RandomGrayscale(p=0.2)(mix_img_s1)
        mix_img_s1 = blur(mix_img_s1, p=0.5)
        # cutmix_box_2 = obtain_cutmix_box(mix_img_s1.size[0], p=0.5)

        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        # img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        # img_s2 = blur(img_s2, p=0.5)
        # cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        # ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, mask = normalize(img_s1, mask)
        mix_img_s1, mix_mask = normalize(mix_img_s1, mix_mask)
        # print(f'img 1 shape: {img_s1.shape}')
        # print(f'mix img 1 shape: {mix_img_s1.shape}')
        # print(f'mask shape:{mask.shape}')

        # cutmix
        mask[cutmix_box == 1] = mix_mask[cutmix_box == 1]
        cutmix_box = repeat(cutmix_box, 'h w -> c h w', c=img_s1.size(0))
        img_s1[cutmix_box == 1] = mix_img_s1[cutmix_box == 1]

        # print(f'box shape 1:{cutmix_box.shape}')
        # print(f'box shape 2:{cutmix_box_2.shape}')

        # img_s2 = normalize(img_s2)

        # mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 254] = 255

        return img_s1, mask

    def __len__(self):
        return len(self.ids)
