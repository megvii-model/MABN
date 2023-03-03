import torch
import torch.distributed as dist


def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_primary():
    return get_rank() == 0


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()