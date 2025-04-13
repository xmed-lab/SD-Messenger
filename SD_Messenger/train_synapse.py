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

from metrics import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from model.model_helper import ModelBuilder

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

    if cfg['dataset'] == 'pascal':
        from dataset.semi_new import SemiDataset
        trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                 cfg['crop_size'], args.unlabeled_id_path)
        trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                 cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
        valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    elif cfg['dataset'] == 'synapse':
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

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)


    diff = Difficulty(cfg['nclass'], accumulate_iters=50)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}/{:.2f} in Epoch {:}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, previous_best_dice, best_epoch))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_consis = AverageMeter()
        # total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)

        for i, ((img_x, mask_x), img_u_s) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_s = img_u_s.cuda()

            with torch.no_grad():
                model.eval()
                pred_u_pseudo, _ = model(img_u_s)
                pred_u_pseudo = pred_u_pseudo.detach()
                pseudo_label = pred_u_pseudo.argmax(dim=1)
            cut_mix_area_list = []

            # L2U transfer
            for b in range(0, img_u_s.shape[0]):
                img_u_s[b, :, :, :], pseudo_label[b, :, :], cut_mix_cor = cutmix_segmentation(img_u_s[b, :, :, :],
                                                                                              pseudo_label[b, :, :],
                                                                                              img_x[b, :, :, :],
                                                                                              mask_x[b, :, :],
                                                                                              cfg['l2u_size'])
                cut_mix_area = torch.zeros([img_u_s.shape[2], img_u_s.shape[3]])
                if torch.sum(cut_mix_cor) > 0:
                    cut_mix_area[cut_mix_cor[0]:cut_mix_cor[2], cut_mix_cor[1]:cut_mix_cor[3]] = 1
                cut_mix_area_list.append(cut_mix_area)

            cut_mix_area = torch.stack(cut_mix_area_list, dim=0).cuda()
            cut_mix_area = repeat(cut_mix_area, 'b h w -> b c h w', c=cfg['nclass'])
            model.train()
            num_lb, num_ulb = img_x.shape[0], img_u_s.shape[0]

            preds, _ = model(torch.cat((img_x, img_u_s)))
            pred_x, pred_u = preds.split([num_lb, num_ulb])

            mask_onehot = torch.nn.functional.one_hot(mask_x, cfg['nclass'])
            mask_onehot = rearrange(mask_onehot, 'b h w c -> b c h w')
            weight_diff = diff.cal_weights(pred_x.detach(), mask_onehot)

            criterion_l.update_weight(weight_diff)

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_w_fp = (criterion_ce(pred_u, pseudo_label) + criterion_dice(pred_u.softmax(dim=1),
                                                                               pseudo_label.unsqueeze(1).float())) / 2.0
            pred_u = pred_u * cut_mix_area
            pred_x = pred_x * cut_mix_area
            loss_consis = torch.nn.functional.mse_loss(pred_x, pred_u)

            if epoch > 5:
                loss = (loss_x + loss_u_w_fp + loss_consis) / 3.0
            else:
                loss = loss_x

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_w_fp.update(loss_u_w_fp.item())
            total_loss_consis.update(loss_consis.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                # writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/loss_consis', loss_consis.item(), iters)
                # writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss w_fp: {:.3f}, Loss consistency: {:.6f}'
                            .format(i, total_loss.avg, total_loss_x.avg, total_loss_w_fp.avg, total_loss_consis.avg))

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

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        previous_best_dice = max(DICE, previous_best_dice)
        if is_best:
            best_epoch = epoch
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
