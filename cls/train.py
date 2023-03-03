import os
import sys
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import time
import logging
from tqdm import tqdm

from utils import accuracy, AvgrageMeter, LabelSmoothCrossEntropyLoss, save_checkpoint, sgd_optimizer, get_train_dataloader, get_val_dataloader
from networks.resnet import resnet50
from stat_util import StatisticsUtil


def get_args():
    parser = argparse.ArgumentParser("ResNet-50 with MABN on Imagenet")

    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=120, help='total epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--save', type=str, default='./MABN', help='path for saving trained models')
    parser.add_argument('--save_interval', type=int, default=2, help='save interval')
    parser.add_argument('--train_dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val_dir', type=str, default='data/val', help='path to validation dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='path to checkpoint')
    parser.add_argument('--test_only', action='store_true', help='if only test the trained model')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_num', type=int, default=8, help='number of gpus')
    parser.add_argument('--record_statistics', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://')

    if args.local_rank == 0:
        log_format = '[%(asctime)s] %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%d %I:%M:%S')
        t = time.time()
        local_time = time.localtime(t)
        if not os.path.exists('{}'.format(args.save)):
            os.makedirs('{}'.format(args.save))
        fh = logging.FileHandler(os.path.join('{}/log.train-{}-{}-{}-{}-{}-{}'.format(args.save, \
                    local_time.tm_year, local_time.tm_mon, local_time.tm_mday, \
                    local_time.tm_hour, local_time.tm_min, local_time.tm_sec)))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        logging.info(args)

    if not args.test_only:
        assert os.path.exists(args.train_dir)
        args.train_dataloader = get_train_dataloader(args.train_dir, \
             args.batch_size//args.gpu_num, args.total_epoch,args.local_rank)

    assert os.path.exists(args.val_dir)
    if args.local_rank == 0:
        args.val_dataloader = get_val_dataloader(args.val_dir)

    print('rank {:d}: load data successfully'.format(args.local_rank))

    model = resnet50()
    optimizer = sgd_optimizer(model, args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    [30, 60, 90, 120],  0.1)

    if args.checkpoint_dir is not None:
        state_dict = torch.load(args.checkpoint_dir, map_location='cpu')
        model.load_state_dict(state_dict['model'])
        if 'optimizer' in state_dict.keys():
            optimizer.load_state_dict(state_dict['optimizer'])
        if 'scheduler' in state_dict.keys():
            scheduler.load_state_dict(state_dict['scheduler'])
        if 'iteration' in state_dict.keys():
            start_epoch = state_dict['iteration']
    else:
        start_epoch = 0

    args.loss_function = LabelSmoothCrossEntropyLoss().cuda()
    device = torch.device("cuda")
    model.to(device)
    for name, param in model.named_parameters():
        if 'momentum_buffer' in optimizer.state[param]:
            optimizer.state[param]['momentum_buffer'] = optimizer.state[param]['momentum_buffer'].cuda() 
        
    model = torch.nn.parallel.DistributedDataParallel(model,  device_ids=[args.local_rank], \
        output_device=args.local_rank, broadcast_buffers=False)

    args.optimizer = optimizer
    args.scheduler = scheduler

    if not args.test_only:
        train(model, device, args, start_epoch=start_epoch+1)

    if args.local_rank == 0:
        validate(model, device, args)

def train(model, device, args, start_epoch):
    optimizer = args.optimizer
    scheduler = args.scheduler
    loss_function = args.loss_function
    train_iterator = iter(args.train_dataloader)

    model.train()
    if args.record_statistics:
        statistic_util = StatisticsUtil()
        statistic_util.bind_bn_statistics(model, batch_size=args.batch_size)

    for i in range(start_epoch, args.total_epoch+1):
        Top1_err, Top5_err, Loss = 0.0, 0.0, 0.0

        if args.local_rank == 0:
            pbar = tqdm(range(5000))

        for iteration in range(5000):
            if args.record_statistics:
                statistic_util.set_step(i * 5000 + iteration)
            data, label = next(train_iterator)
            data = data.cuda()
            target = label.type(torch.long).cuda()

            output = model(data)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            Loss += loss.item()
            Top1_err += 100 - prec1.item()
            Top5_err += 100 - prec5.item()

            if args.local_rank == 0:
                pbar.update()

        scheduler.step()

        if  args.local_rank == 0:
            printInfo = 'TRAIN Epoch {}: lr = {:.6f},\tloss = {:.6f},\t'.format(i, scheduler.get_lr()[0], Loss/5000) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err/5000) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err/5000)
            logging.info(printInfo)

        if i % args.save_interval == 0 and args.local_rank == 0:
            save_checkpoint(model, i, args.save, optimizer, scheduler)


def validate(model, device, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataloader = args.val_dataloader
    L = len(val_dataloader)

    model.eval()
    with torch.no_grad():
        data_iterator = enumerate(val_dataloader)
        for _ in tqdm(range(250)):
            _, data = next(data_iterator)
            target = data[1].type(torch.LongTensor)
            data, target = data[0].to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item())
            top1.update(prec1.item())
            top5.update(prec5.item())

    if args.local_rank == 0:
        logInfo = 'TEST: loss = {:.6f},\t'.format(objs.avg) + \
                  'Top-1 err = {:.6f},\t'.format(100 - top1.avg) + \
                  'Top-5 err = {:.6f},\t'.format(100 - top5.avg)
        logging.info(logInfo)

if __name__ == "__main__":
    main()

