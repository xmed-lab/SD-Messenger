import argparse
import logging
import os
import pprint
import warnings
import pandas as pd
import cv2
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from model.semseg.segformerhead import SegFormer
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

parser = argparse.ArgumentParser(description='S&D Messenger: Exchanging Semantic and Domain Knowledge for Generic Semi-Supervised Medical Image Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


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

    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']

    with torch.no_grad():
        for img, mask, id in tqdm(loader):
            img = img.cuda()
            if mode == 'center_crop':
                h, w = img.shape[-2:]
                start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
            h, w = img.shape[-2:]
            pred, all_spark_feat = model(img)
            pseudo_prob_map = pred
            # pred = pred.argmax(dim=1)
            # print(f'prob map shape: {pseudo_prob_map.shape}')
            # print(f'feature map shape: {all_spark_feat.shape}')
            h_, w_ = all_spark_feat.shape[-2:]
            pseudo_prob_map = F.interpolate(pseudo_prob_map, size=(h_, w_), mode="bilinear", align_corners=True)
            pseudo_prob_map = torch.flatten(pseudo_prob_map, start_dim=2, end_dim=-1)
            all_spark_feat = torch.flatten(all_spark_feat, start_dim=2, end_dim=-1).permute(0, 2, 1).contiguous()
            # print(f'prob map shape after: {pseudo_prob_map.shape}')
            # print(f'feature map shape after: {all_spark_feat.shape}')
            sim_matrix = torch.matmul(F.normalize(pseudo_prob_map, dim=-1), F.normalize(all_spark_feat, dim=1))[0, :, :]
            sim_matrix = F.normalize(sim_matrix, dim=0)
            sim_index = sim_matrix.argmax(dim=0).cpu().numpy()

            sorted_indices = np.argsort(sim_index)
            sorted_sim_matrix = sim_matrix.cpu().numpy()[:, sorted_indices]
            # print(sim_matrix)
            import matplotlib.pyplot as plt

            labels = []
            file_name = os.path.basename(id[0].split(' ')[0])
            os.makedirs(f'sim/{file_name}', exist_ok=True)
            fig, ax = plt.subplots(figsize=(512, 21))
            cax = ax.matshow(sorted_sim_matrix, interpolation='nearest')

            ax.grid(True)
            plt.xticks(range(512), labels, rotation=90)
            plt.yticks(range(21), labels)
            # fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
            plt.show()
            plt.savefig(f'sim/{file_name}/sim.jpg')
            channel_count_list = []
            for value in range(21):
                count = np.count_nonzero(sim_index == value)
                channel_count_list.append(count)
            # print(sim_matrix)

            np.savetxt(f'sim/{file_name}/channel_count.csv', np.array(channel_count_list), delimiter=",")

            copy_img = Image.open(os.path.join('VOC2012', id[0].split(' ')[0]))
            copy_img.save(f'sim/{file_name}/original.jpg')
            # file_name = os.path.basename(id[0].split(' ')[0])
            # print(f'id: {file_name}')
            # os.makedirs(f'pred', exist_ok=True)
            # pred = np.where(pred > 0, 255, 0)
            # print(pred.shape)
            # cv2.imwrite(f'pred/{file_name}', pred_[0, :, :])

            # os.makedirs(f'feat/{file_name}', exist_ok=True)
            # for c in range(512):
            #     copy_img = Image.open(os.path.join('VOC2012', id[0].split(' ')[0]))
            #     memory_bank_i_c = all_spark_feat[0, c, :, :].cpu().numpy()
            #     # plt.imshow(memory_bank_i_c, cmap='jet')
            #     # img_plot = plt.imshow(memory_bank_i_c)
            #
            #     # print(f'save feat/{id}/{c}.jpg')
            #     # plt.savefig(f'feat/cls_{cls}/memory_bank_fm_{cls}_{c}.jpg')
            #     copy_img.save(f'feat/{file_name}/original.jpg')
            #     plt.imsave(f'feat/{file_name}/{c}.jpg', memory_bank_i_c, cmap='jet')
            #     original_img = cv2.imread(f'feat/{file_name}/original.jpg')
            #     feat = cv2.imread(f'feat/{file_name}/{c}.jpg')
            #     alpha = 0.3
            #     vis_img = original_img * (1 - alpha) + feat * alpha
            #     cv2.imwrite(f'feat/{file_name}/{c}_vis.jpg', vis_img)


    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    idx = 0
    values = np.zeros((sample_number, len(class_list), 4))
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
                # print(pred.shape)
                pred = pred.argmax(dim=1).cpu().numpy()[0, :, :]
                mask = mask.cpu().numpy()[0, :, :]

            for i in range(0, len(class_list)):
                pred_i = (pred == i)
                label_i = (mask == i)
                if pred_i.sum() == 0 and label_i.sum() == 0:
                    dice, jaccard, hd95, asd = 0, 0, 0, 0
                    values[idx][i] = np.array([dice, jaccard, hd95, asd])
                else:
                    dice = metric.binary.dc(pred == i, mask == i)
                    jaccard = metric.binary.jc(pred == i, mask == i)
                    # hd95 = metric.binary.hd95(pred == i, mask == i)
                    # asd = metric.binary.asd(pred == i, mask == i)
                    values[idx][i] = np.array([dice, jaccard, 0, 0])
                # elif pred_i.sum() > 0 and label_i.sum() == 0:
                #     dice, jaccard, hd95, asd = 0, 0, 128, 128
                # elif pred_i.sum() == 0 and label_i.sum() > 0:
                #     dice, jaccard, hd95, asd = 0, 0, 128, 128
                # elif pred_i.sum() == 0 and label_i.sum() == 0:
                #     dice, jaccard, hd95, asd = 1, 1, 0, 0
            # for i in range(0, len(class_list)):
            #     pred_i = (pred == i).astype(int)
            #     # print(pred_i)
            #     label_i = (mask == i).astype(int)
            #
            #
            #     if pred_i.sum() > 0 and label_i.sum() > 0:
            #         dice = metric.binary.dc((pred == i).astype(int), (mask == i).astype(int)) * 100
            #         asd = metric.binary.asd((pred == i).astype(int), (mask == i).astype(int))
            #         hd95 = metric.binary.hd95((pred == i).astype(int), (mask == i).astype(int))
            #         jaccard = metric.binary.jc((pred == i).astype(int), (mask == i).astype(int)) * 100
            #         values[idx, i, :] = np.array([dice, jaccard, hd95, asd])
            #         # values[idx][i-1] = np.array([dice, hd95])
            #     elif pred_i.sum() > 0 and label_i.sum() == 0:
            #         dice, jaccard, hd95, asd = 0, 0, 128, 128
            #     elif pred_i.sum() == 0 and label_i.sum() > 0:
            #         dice, jaccard, hd95, asd = 0, 0, 128, 128
            #     elif pred_i.sum() == 0 and label_i.sum() == 0:
            #         dice, jaccard, hd95, asd = 1, 1, 0, 0
            #     # print(values[idx, i, :])
            #     values[idx, i, :] = np.array([dice, jaccard, hd95, asd])
                # print(values[idx, i, :])
            idx = idx + 1
            # values_for_each_case = np.mean(values[idx, :, :], axis=1)
            # print(f'image: {id}, number: {idx}, Dice: {values_for_each_case[0]} jaccard: {values_for_each_case[1]} hd95: {values_for_each_case[2]} asd: {values_for_each_case[3]} ')
        values_for_each_classes = np.mean(values, axis=0)
        return values_for_each_classes

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
