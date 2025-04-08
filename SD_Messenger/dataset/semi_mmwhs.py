from dataset.strong_aug import get_StrongAug, CenterCrop, ToTensor
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
import cv2


class MMWHSDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        print(id_path)
        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            split = id_path.replace('splits/mmwhs/', '').replace('/labeled.txt', '')
            print(split)
            with open(f'splits/%s/test_{split}.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

        self.transforms_train_labeled = get_StrongAug((self.size, self.size), 3, 0.7)
        self.transforms_train_unlabeled = get_StrongAug((self.size, self.size), 3, 0.7)

    def __getitem__(self, item):
        id = self.ids[item]

        # img = cv2.imread(os.path.join(self.root, id.split(' ')[0]), 0)
        # mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]), 0)
        # print(os.path.join(self.root, id.split(' ')[0]))
        img = np.load(os.path.join(self.root, id.split(' ')[0])).astype(np.float32)
        mask = np.load(os.path.join(self.root, id.split(' ')[1])).astype(np.int8)

        # p5 = np.percentile(img.flatten(), 0.5)
        # p95 = np.percentile(img.flatten(), 99.5)
        # # print(p5, p95)
        # img = img.clip(min=p5, max=p95)
        # # normalize
        img = (img - img.min()) / (img.max() - img.min())

        if self.mode == 'val':
            # already normalized
            sample = {'image': img, 'label': mask}

            sample = transforms.Compose([
                CenterCrop((self.size, self.size)),
                ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(sample)

            img, mask = sample['image'], sample['label']
            img = repeat(img, 'c h w -> (repeat c) h w', repeat=3)
            return img, mask, id

        if self.mode == 'train_u':
            # normalize
            img = img.astype(np.float32)
            mask = mask.astype(np.int8)
            sample = {'image': img, 'label': mask}

            # contains numpy to tensor
            sample = self.transforms_train_unlabeled(sample)
            img = repeat(sample['image'], 'c h w -> (repeat c) h w', repeat=3)
            return img

        elif self.mode == 'train_l':
            img_s1 = deepcopy(img)

            img_s1 = img_s1.astype(np.float32)
            mask = mask.astype(np.int8)
            sample = {'image': img_s1, 'label': mask}

            # contains numpy to tensor
            sample = self.transforms_train_labeled(sample)
            img = repeat(sample['image'], 'c h w -> (repeat c) h w', repeat=3)

            return img, sample['label'].long()

    def __len__(self):
        return len(self.ids)
