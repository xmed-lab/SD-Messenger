import argparse
import csv
import logging
import os
import pprint
import random
import warnings
import pandas as pd
import cv2
import torch
import numpy as np
from numpy import interp
from sklearn.manifold import TSNE
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_score

from model.model_helper import ModelBuilder
from dataset.semi import SemiDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from medpy import metric
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, f1_score

parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


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
    alpha = 1.0
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

def pixelError(output, target, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)

    output[np.where(target == ignore_index)[0]] = ignore_index

    error_area = np.where(output != target, 1, 0)
    summary_area = np.where(output != ignore_index, 1, 0)

    pixel_error = np.sum(error_area) / np.sum(summary_area)

    return np.expand_dims(pixel_error, axis=0)


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    pixel_error_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred, _ = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)
                pred = final.argmax(dim=1)
            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                pred, _ = model(img)
                pred = pred.argmax(dim=1)

            # mIou metrics
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            pixel_error = pixelError(pred.cpu().numpy(), mask.numpy())

            reduced_pixel_error = torch.from_numpy(pixel_error).cuda()

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_pixel_error)
            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            pixel_error_meter.update(reduced_pixel_error.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    dice_class = (2 * (iou_class / 100.0)) / (1 + (iou_class / 100.0)) * 100.0
    mIOU = np.mean(iou_class[1:])
    DICE = np.mean(dice_class[1:])
    mean_error = pixel_error_meter.avg

    return mIOU, iou_class, DICE
