from dataset.strong_aug import get_StrongAug
from dataset.transform import *
from einops import rearrange, repeat
from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.augmentation_rgb import *
import cv2


class SynapseDataset(Dataset):
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
            with open('splits/%s/val_rgb.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

        self.transforms_train_labeled = get_StrongAug((self.size, self.size), 3, 0.7)
        self.transforms_train_unlabeled = get_StrongAug((self.size, self.size), 3, 0.7)

    def __getitem__(self, item):
        id = self.ids[item]

        img = cv2.imread(os.path.join(self.root, id.split(' ')[0]), 1)
        mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]), 0)

        if self.mode == 'val':
            # already normalized
            img = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(img)

            # img = repeat(img, 'c h w -> (repeat c) h w', repeat=3)
            return img, mask, id

        if self.mode == 'train_u':
            # normalize
            spatial_transform = DualCompose([
                RandomFlip(),
                Rotate(),
            ])

            img = img.astype(np.float32)
            mask = mask.astype(np.int8)

            img, mask = spatial_transform(img, mask)
            # img = pixel_transform(img)
            # sample = {'image': img, 'label': mask}

            # contains numpy to tensor
            # sample = self.transforms_train_unlabeled(sample)

            # img = repeat(img, 'c h w -> (repeat c) h w', repeat=3)
            return img
        elif self.mode == 'train_l':
            img_s1 = deepcopy(img)
            spatial_transform = DualCompose([
                RandomFlip(),
                Rotate(),
            ])

            # normalize
            img_s1 = img_s1.astype(np.float32)
            # print(f'before trans: {img_s1.max()}')
            # print(f'img shape: {img_s1.shape}')
            mask = mask.astype(np.int8)
            img_s1, mask = spatial_transform(img_s1, mask)
            # sample = {'image': img_s1, 'label': mask}

            # contains numpy to tensor
            # sample = self.transforms_train_labeled(sample)

            # img = repeat(sample['image'], 'c h w -> (repeat c) h w', repeat=3)
            # print(f'mask shape: {mask.shape}')
            # print(f'after trans: {img_s1.max()}')
            return img, torch.from_numpy(mask).long()

    def __len__(self):
        return len(self.ids)
