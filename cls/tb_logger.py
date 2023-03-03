import torch

from torch.utils.tensorboard import SummaryWriter
from dist import is_primary


class Logger(object):
    def __init__(self, log_dir=None) -> None:
        self.log_dir = log_dir
        if SummaryWriter is not None and is_primary():
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None

    def log_scalars(self, scalar_dict, step, prefix=None):
        if self.writer is None:
            return
        for k in scalar_dict:
            v = scalar_dict[k]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            if prefix is not None:
                k = prefix + k
            self.writer.add_scalar(k, v, step)