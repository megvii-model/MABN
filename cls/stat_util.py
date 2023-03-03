import torch
import torch.nn as nn
import torchvision.models as models

import collections
from functools import partial
from tb_logger import Logger
from dist import get_world_size


class StatisticsUtil:

    def __init__(self, log_dir="./output"):
        self.log_dir = log_dir
        self.logger = None
        self.buffer = {}
        self.step = 0

    def set_step(self, step):
        self.step = step

    def _forward_hook(self, name, module, x, y, channel_idx=0):
        x, y = x[0], y[0]
        mu_x = x.mean(dim=-1).mean(dim=-1).mean(dim=0)
        mu_x2 = (x * x).mean(dim=-1).mean(dim=-1).mean(dim=0)
        var_x = mu_x2 - mu_x * mu_x
        self.buffer[name]["y"] = y
        self.logger.log_scalars({
            "mu_x": mu_x[channel_idx],
            "var_x": var_x[channel_idx],
        }, step=self.step, prefix=name + "_")

    def _backward_hook(self, name, module, grad_x, grad_y, channel_idx=0):
        grad_x, grad_y = grad_x[0], grad_y[0]
        mu_g = grad_y.mean(dim=-1).mean(dim=-1).mean(dim=0)
        mu_g = mu_g * module.weight
        mu_gy = (grad_y * self.buffer[name]["y"]).mean(dim=-1).mean(dim=-1).mean(dim=0)
        self.logger.log_scalars({
            "mu_g": mu_g[channel_idx],
            "mu_gy": mu_gy[channel_idx]
        }, step=self.step, prefix=name + "_")

    def bind_bn_statistics(self, model, batch_size=32):
        self.logger = Logger(log_dir=self.log_dir + f"/bs_{batch_size // get_world_size()}")
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.buffer[name] = dict()
                module.register_forward_hook(partial(self._forward_hook, name))
                module.register_backward_hook(partial(self._backward_hook, name))



