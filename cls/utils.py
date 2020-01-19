import os
import re
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from torch.nn.modules.loss import _Loss
import PIL
from PIL import Image
from torchvision import transforms, datasets
import torch.nn.functional as F
import cv2
from torch.utils.data import Sampler

import random
import math

from SGD import SGD


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def save_checkpoint(model, iters, path, optimizer=None, scheduler=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print("Saving checkpoint to file {}".format(path))
    state_dict = {}
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        key = k
        if k.split('.')[0] == 'module':
              key = k[7:]
        new_state_dict[key] = v
    state_dict['model'] = new_state_dict

    state_dict['iteration'] = iters
    if optimizer is not None:
          state_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
          state_dict['scheduler'] = scheduler.state_dict()

    filename = os.path.join("{}/checkpoint.pth".format(path))
    try:
        torch.save(state_dict, filename)
    except:
        print('save {} failed, continue training'.format(path))

def sgd_optimizer(model, base_lr, momentum, weight_decay):
    params = []
    for key, value in model.named_parameters():
        params.append(value)
    param_group = [{'params': params,
                    'weight_decay': weight_decay}]
    optimizer = SGD(param_group, lr = base_lr, momentum=momentum)
    return optimizer

## data augmentation functions

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:,::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:,::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img


class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:,::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


class RandomResizedCrop(object):

    def __init__(self, scale=(0.08, 1.0), target_size:int=224, max_attempts:int=10):
        assert scale[0] <= scale[1]
        self.scale = scale
        assert target_size > 0
        self.target_size = target_size
        assert max_attempts >0
        self.max_attempts = max_attempts

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.uint8)
        H, W, C = img.shape
        
        well_cropped = False
        for _ in range(self.max_attempts):
            crop_area = (H*W) * random.uniform(self.scale[0], self.scale[1])
            crop_edge = round(math.sqrt(crop_area))
            dH = H - crop_edge
            dW = W - crop_edge
            crop_left = random.randint(min(dW, 0), max(dW, 0))
            crop_top = random.randint(min(dH, 0), max(dH, 0))
            if dH >= 0 and dW >= 0:
                well_cropped = True
                break
        
        crop_bottom = crop_top + crop_edge
        crop_right = crop_left + crop_edge
        if well_cropped:
            crop_image = img[crop_top:crop_bottom,:,:][:,crop_left:crop_right,:]
            
        else:
            roi_top = max(crop_top, 0)
            padding_top = roi_top - crop_top
            roi_bottom = min(crop_bottom, H)
            padding_bottom = crop_bottom - roi_bottom
            roi_left = max(crop_left, 0)
            padding_left = roi_left - crop_left
            roi_right = min(crop_right, W)
            padding_right = crop_right - roi_right

            roi_image = img[roi_top:roi_bottom,:,:][:,roi_left:roi_right,:]
            crop_image = cv2.copyMakeBorder(roi_image, padding_top, padding_bottom, padding_left, padding_right,
                borderType=cv2.BORDER_CONSTANT, value=0)
            
        target_image = cv2.resize(crop_image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        target_image = PIL.Image.fromarray(target_image.astype('uint8'))
        return target_image


class LighteningJitter(object):

    def __init__(self, eigen_vecs, eigen_values, max_eigen_jitter=0.1):

        self.eigen_vecs = np.array(eigen_vecs, dtype=np.float32)
        self.eigen_values = np.array(eigen_values, dtype=np.float32)
        self.max_eigen_jitter = max_eigen_jitter

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.float32)
        img = np.ascontiguousarray(img/255)

        cur_eigen_jitter = np.random.normal(scale=self.max_eigen_jitter, size=self.eigen_values.shape)
        color_purb = (self.eigen_vecs @ (self.eigen_values * cur_eigen_jitter)).reshape([1, 1, -1])
        img += color_purb
        img = np.ascontiguousarray(img*255)
        img.clip(0, 255, out=img)
        img = PIL.Image.fromarray(np.uint8(img))
        return img

class Random_Batch_Sampler(Sampler):

    def __init__(self, dataset, batch_size, total_iters, rank=None):
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")

        self.dataset_num = dataset.__len__()
        self.rank = rank
        self.batch_size = batch_size
        self.total_iters = total_iters


    def __iter__(self):
        random.seed(self.rank)
        for i in range(self.total_iters):
            batch_iter = []
            for _ in range(self.batch_size):
                batch_iter.append(random.randint(0, self.dataset_num-1))
            
            yield batch_iter

    def __len__(self):
        return self.total_iters

class LabelSmoothCrossEntropyLoss(_Loss):

    def __init__(self, eps=0.1, class_num=1000):
        super(LabelSmoothCrossEntropyLoss, self).__init__()

        self.min_value = eps / class_num
        self.eps = eps

    def __call__(self, pred:torch.Tensor, target:torch.Tensor):

        epses = self.min_value * torch.ones_like(pred)
        log_probs = F.log_softmax(pred, dim=1)

        if target.ndimension() == 1:
            target = target.expand(1, *target.shape)
            target = target.transpose(1, 0)
        target = torch.zeros_like(log_probs).scatter_(1, target, 1)
        target = target.type(torch.float)
        target = target * (1 - self.eps) + epses

        element_wise_mul = log_probs * target * -1.0

        loss = torch.sum(element_wise_mul, 1)
        loss = torch.mean(loss)

        return loss

def get_train_dataloader(train_dir, batch_size, total_iters,local_rank):
    eigvec = np.array([
        [-0.5836, -0.6948,  0.4203],
        [-0.5808, -0.0045, -0.8140],
        [-0.5675,  0.7192,  0.4009]
    ])
    eigval = np.array([0.2175, 0.0188, 0.0045])

    train_dataset = datasets.ImageFolder(train_dir,
        transforms.Compose([
            RandomResizedCrop(target_size=224, scale=(0.08, 1.0)),
            LighteningJitter(eigen_vecs=eigvec[::-1,:], eigen_values=eigval, max_eigen_jitter=0.1),
            transforms.RandomHorizontalFlip(0.5),
            ToBGRTensor(),
        ])
    )

    datasampler = Random_Batch_Sampler(
        train_dataset, batch_size=batch_size,
        total_iters=total_iters*50000, rank=local_rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=8,
        pin_memory=True, batch_sampler=datasampler)

    return train_loader

def get_val_dataloader(val_dir):
    val_dataset = datasets.ImageFolder(train_dir, 
        transforms.Compose([
                OpencvResize(256),
                transforms.CenterCrop(224),
                ToBGRTensor(),
            ]))
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=200, shuffle=False,
            num_workers=8, pin_memory=True
        )

    return val_loader 
