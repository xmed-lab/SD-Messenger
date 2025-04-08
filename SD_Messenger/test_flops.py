import argparse
import logging
import os
import pprint

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from einops import repeat, rearrange
import yaml

from dataset.cutmix_augmentation import cutmix_segmentation
from util.loss import DC_and_CE_loss, SoftDiceLoss
from util.acdc_utils import DiceLoss
#
# from dataset.semi_new import SemiDataset
# from dataset.semi_cityscapes import SemiDataset
# from model.semseg.crosstr import SegFormer
from supervised import evaluate, evaluate_med
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from model.model_helper import ModelBuilder
import torch
import torchvision.models as models
from thop import profile
from thop import clever_format
import time

parser = argparse.ArgumentParser(
    description='S&D Messenger: Exchanging Semantic and Domain Knowledge for Generic Semi-Supervised Medical Image Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=1234, type=int)
parser.add_argument('--seed', default=42, type=int)


def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight


class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred, label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice > 0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice <= 0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)

        self.last_dice = cur_dice

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn,
                             momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn,
                               momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)

        cur_diff = torch.pow(cur_diff, 1 / 5)

        self.dice_weight = EMA(1. - cur_dice, self.dice_weight,
                               momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        # weights = weights / weights.max()
        return weights * self.num_cls


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def __init__(self, weight=None):
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight)

    def forward(self, input, target):
        return super().forward(input, target)

    def update_weight(self, weight):
        self.weight = weight


def main():
    args = parser.parse_args()
    import random
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = ModelBuilder(cfg['model'])
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = RobustCrossEntropyLoss().cuda(local_rank)
        criterion_ce = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
        criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    if cfg['dataset'] == 'synapse':
        from dataset.semi_synapse_aug import SynapseDataset
        trainset_u = SynapseDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                    cfg['crop_size'], args.unlabeled_id_path)
        trainset_l = SynapseDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                    cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
        valset = SynapseDataset(cfg['dataset'], cfg['data_root'], 'val')
    else:
        raise NotImplementedError('%s dataset is not implemented' % cfg['dataset'])

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    previous_best_dice = 0.0
    best_epoch = 0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'best.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class, DICE = evaluate(model, valloader, eval_mode, cfg)

    if rank == 0:
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
    
        for (cls_idx, iou) in enumerate(iou_class):
            dice = (2 * (iou / 100) / (1 + (iou / 100))) * 100
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'DICE: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
        logger.info('***** Evaluation {} ***** >>>> Mean DICE: {:.2f}\n'.format(eval_mode, DICE))

if __name__ == '__main__':
    main()
