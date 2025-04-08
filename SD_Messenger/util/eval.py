import numpy as np
import torch
from medpy import metric
import argparse
import logging
import os
import pprint
import warnings
import pandas as pd
import cv2
import torch
import numpy as np
import torch.distributed as dist
from utils import AverageMeter, intersectionAndUnion
from medpy import metric


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

            # ASD metrics

            # 95HD metrics

            # Jaccard

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    dice_class = (2 * (iou_class / 100.0)) / (1 + (iou_class / 100.0)) * 100.0
    mIOU = np.mean(iou_class[1:])
    DICE = np.mean(dice_class[1:])
    mean_error = pixel_error_meter.avg

    return mIOU, iou_class, DICE


def evaluate_med(model, loader, mode, cfg, class_list, sample_number):
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
