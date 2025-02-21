import torch
from math import cos, pi


"""
    modified from 
    https://github.com/thangvubk/SoftGroup/blob/a87940664d0af38a3f55cdac46f86fc83b029ecc/softgroup/util/utils.py#L55
"""
def cosine_lr_decay(optimizer, base_lr, current_epoch, start_epoch, total_epochs, clip):
    if current_epoch < start_epoch:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] = clip + 0.5 * (base_lr - clip) * \
                            (1 + cos(pi * ((current_epoch - start_epoch) / (total_epochs - start_epoch))))

def adjust_learning_rate(
    initial_lr, optimizer, num_iterations, total_iterations, decreased_by
):
    adjust_lr_every = int(total_iterations / 2)
    lr = initial_lr * ((1 / decreased_by) ** (num_iterations // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr