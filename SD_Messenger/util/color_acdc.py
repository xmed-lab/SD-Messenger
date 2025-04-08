import logging
import os
import time
from argparse import ArgumentParser
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from PIL import Image
from tqdm import tqdm
import glob


def main():
    colormap = create_pascal_label_colormap()
    os.makedirs('../ACDC/color_mask', exist_ok=True)
    os.makedirs('../ACDC/masked_img', exist_ok=True)
    for label_path in tqdm(glob.glob(os.path.join("../ACDC/mask/", '*.png'))):
        img_path = os.path.join('../ACDC/img', os.path.basename(label_path))
        masked_img_path = os.path.join('../ACDC/masked_img', os.path.basename(label_path))
        # print(label_path)
        mask = np.array(Image.open(label_path))
        if 'original' in label_path:
            color_mask = Image.fromarray(mask)
            color_mask.save(label_path.replace("pred_uni_t", "pred_color_uni_t"))
            continue
        # print(mask.shape)

        color_mask = Image.fromarray(colorful(mask, colormap))
        color_mask.save(label_path.replace("mask", "color_mask"))

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = np.zeros_like(img)
        img_rgb[:, :, 0] = gray
        img_rgb[:, :, 1] = gray
        img_rgb[:, :, 2] = gray

        color_mask = colorful(mask, colormap)

        alpha = 0.7

        img_rgb = np.where(color_mask, img_rgb * (1 - alpha) + color_mask * alpha, img_rgb)
        cv2.imwrite(masked_img_path, img_rgb)




def colorful(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return np.uint8(color_mask)


def create_pascal_label_colormap():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    colormap[19] = [128, 192, 0]
    colormap[20] = [0, 64, 128]

    return colormap


if __name__ == "__main__":
    main()
