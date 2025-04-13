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

from supervised import evaluate, evaluate_med
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from model.model_helper import ModelBuilder

parser = argparse.ArgumentParser(
    description='S&D Messenger: Exchanging Semantic and Domain Knowledge for Generic Semi-Supervised Medical Image Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=1234, type=int)
parser.add_argument('--seed', default=42, type=int)

def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight

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

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['dataset'] == 'pascal':
        from dataset.semi_new import SemiDataset
        trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                 cfg['crop_size'], args.unlabeled_id_path)
        trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                 cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
        valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    elif cfg['dataset'] == 'synapse':
        from dataset.semi_synapse_aug import SynapseDataset
        valset = SynapseDataset(cfg['dataset'], cfg['data_root'], 'val')
    else:
        raise NotImplementedError('%s dataset is not implemented' % cfg['dataset'])

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    epoch = -1

    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    model.eval()
    mIoU, iou_class, DICE = evaluate(model, valloader, 'original', cfg)

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
