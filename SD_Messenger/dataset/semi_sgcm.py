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

def vis_save(original_img, pred, save_path):
    blue = [30, 144, 255]  # aorta
    green = [0, 255, 0]  # gallbladder
    red = [255, 0, 0]  # left kidney
    cyan = [0, 255, 255]  # right kidney
    pink = [255, 0, 255]  # liver
    yellow = [255, 255, 0]  # pancreas
    purple = [128, 0, 255]  # spleen
    orange = [255, 128, 0]  # stomach
    color_1 = [128, 0, 0]
    color_2 = [128, 128, 0]
    color_3 = [64, 0, 0]
    color_4 = [64, 128, 0]
    color_5 = [192, 0, 0]

    original_img = original_img
    original_img = original_img.astype(np.uint8)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
    alpha = 0.7
    original_img = np.where(pred == 1, alpha * np.full_like(original_img, blue) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 2, alpha * np.full_like(original_img, green) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 3, alpha * np.full_like(original_img, red) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 4, alpha * np.full_like(original_img, cyan) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 5, alpha * np.full_like(original_img, pink) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 6, alpha * np.full_like(original_img, yellow) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 7, alpha * np.full_like(original_img, purple) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 8, alpha * np.full_like(original_img, orange) + (1 - alpha) * original_img,
                            original_img)

    original_img = np.where(pred == 9, alpha * np.full_like(original_img, color_1) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 10, alpha * np.full_like(original_img, color_2) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 11, alpha * np.full_like(original_img, color_3) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 12, alpha * np.full_like(original_img, color_4) + (1 - alpha) * original_img,
                            original_img)
    original_img = np.where(pred == 13, alpha * np.full_like(original_img, color_5) + (1 - alpha) * original_img,
                            original_img)
    print(f' max: {original_img.max()}, min: {original_img.min()}')
    original_img = cv2.cvtColor(original_img.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)

class SGCMDataset(Dataset):
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
            split = id_path.replace('splits/sgcm/', '').replace('/labeled.txt', '').replace('DG_', '')
            print(split)
            with open(f'splits/%s/test_{split}.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

        self.transforms_train_labeled = get_StrongAug((288, 288), 3, 0.7)
        self.transforms_train_unlabeled = get_StrongAug((288, 288), 3, 0.7)

    def __getitem__(self, item):
        id = self.ids[item]

        # img = cv2.imread(os.path.join(self.root, id.split(' ')[0]), 0)
        # mask = cv2.imread(os.path.join(self.root, id.split(' ')[1]), 0)
        # print(os.path.join(self.root, id.split(' ')[0]))
        img = np.load(os.path.join(self.root, id.split(' ')[0]))
        mask = np.load(os.path.join(self.root, id.split(' ')[1]))
        img = img['arr_0']
        mask = mask['arr_0']

        # print(f'before img minmax: {img.min()}, {img.max()}, mask minmax: {mask.min()}, {mask.max()}')
        # print(f'img shape: {img.shape}, mask shape: {mask.shape}')
        mask = mask[:, :, 1].astype(np.int8)
        img = img.astype(np.float32)
        # print(os.path.join('sgcm_vis', os.path.basename(id.split(' ')[0].replace('npz', 'png'))))

        # cv2.imwrite(os.path.join('sgcm_vis', os.path.basename(id.split(' ')[0].replace('npz', 'png'))), img * 255)
        # vis_save(img * 255, mask, os.path.join('sgcm_vis', os.path.basename(id.split(' ')[0].replace('npz', 'png'))))
        # print(f'mask type: {np.unique(mask)}')
        # print(f'img shape: {img.shape}, mask shape: {mask.shape}')
        # print(f'after img minmax: {img.min()}, {img.max()}, mask minmax: {mask.min()}, {mask.max()}')
        # p5 = np.percentile(img.flatten(), 0.5)
        # p95 = np.percentile(img.flatten(), 99.5)
        # # print(p5, p95)
        # img = img.clip(min=p5, max=p95)
        # # normalize
        # img = (img - img.min()) / (img.max() - img.min())

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
            # sample = self.transforms_train_unlabeled(sample)
            # img = sample['image']
            sample = transforms.Compose([
                CenterCrop((self.size, self.size)),
                ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(sample)
            # print(f'before img minmax: {img.min()}, {img.max()}, mask minmax: {mask.min()}, {mask.max()}')
            img = repeat(sample['image'], 'c h w -> (repeat c) h w', repeat=3)
            return img

        elif self.mode == 'train_l':
            img_s1 = deepcopy(img)

            img_s1 = img_s1.astype(np.float32)
            mask = mask.astype(np.int8)
            sample = {'image': img_s1, 'label': mask}

            # contains numpy to tensor
            # sample = self.transforms_train_labeled(sample)
            sample = transforms.Compose([
                CenterCrop((self.size, self.size)),
                ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])(sample)
            img = repeat(sample['image'], 'c h w -> (repeat c) h w', repeat=3)

            return img, sample['label'].long()

    def __len__(self):
        return len(self.ids)
