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
    # colormap = create_pascal_label_colormap()
    os.makedirs('../synapse_vis/vis_mask', exist_ok=True)
    os.makedirs('../synapse_vis/masked_img', exist_ok=True)
    os.makedirs('../synapse_vis/masked_img_trans', exist_ok=True)
    for label_path in tqdm(glob.glob(os.path.join("../../UniMatch/synapse/synapse_anno/", '*.png'))):
        img_path = os.path.join("../../UniMatch/synapse/synapse_or/", os.path.basename(label_path)).replace('label',
                                                                                                            'img')
        masked_img_path = os.path.join('../synapse_vis/masked_img_trans', os.path.basename(label_path))
        img_gray = cv2.imread(img_path, 0)
        mask = cv2.imread(label_path, 0)
        vis_save(img_gray, mask, masked_img_path)
        # vis_save(np.zeros_like(img_gray), mask, os.path.join('../synapse_vis/vis_mask', os.path.basename(label_path)))
        # # print(label_path)
        # mask = np.array(Image.open(label_path))
        # # if 'original' in label_path:
        # #     color_mask = Image.fromarray(mask)
        # #     color_mask.save(label_path.replace("pred_uni_t", "pred_color_uni_t"))
        # #     continue
        # # print(mask.shape)
        #
        # color_mask = Image.fromarray(colorful(mask, colormap))
        # color_mask.save(os.path.join('../synapse_vis/vis_mask', os.path.basename(label_path)))
        #
        # img_rgb = cv2.imread(img_path, 1)
        # print(f'min: {img_rgb.min()}, max: {img_rgb.max()}')
        # # gray = np.array(Image.open(img_path))
        # # img_rgb = img_rgb.clip(min=0, max=255).astype(np.uint8)
        #
        # # img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        #
        # color_mask = colorful(mask, colormap)
        # alpha = 0.5
        #
        # img_rgb = np.where(color_mask, color_mask, img_rgb)
        # # img_rgb = Image.fromarray(img_rgb)
        # # img_rgb.save(masked_img_path)
        # cv2.imwrite(masked_img_path, img_rgb)


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
    alpha = 0.2
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
    # print(f' max: {original_img.max()}, min: {original_img.min()}')
    original_img = cv2.cvtColor(original_img.astype(np.float32), cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, original_img)


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
    # colormap[14] = [64, 128, 128]
    # colormap[15] = [192, 128, 128]
    # colormap[16] = [0, 64, 0]
    # colormap[17] = [128, 64, 0]
    # colormap[18] = [0, 192, 0]
    # colormap[19] = [128, 192, 0]
    # colormap[20] = [0, 64, 128]

    return colormap


if __name__ == "__main__":
    main()
