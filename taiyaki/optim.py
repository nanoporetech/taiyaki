# Code used by training scripts.

import math
import torch


class ReciprocalLR(torch.optim.lr_scheduler._LRScheduler):
    """Pytorch learning rate scheduler.
    Learning rate is the base learning rate
    (set in the optimiser), multiplied by
    
    1 / (1 + niter/lr_decay_iterations),
    
    so that after lr_decay iterations, the l.r. is divided by 2.
    
    Based on
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """
    def __init__(self, optimizer, lr_decay_iterations, last_epoch=-1):
        self.lr_decay_iterations = lr_decay_iterations
        super(ReciprocalLR, self).__init__(optimizer, last_epoch)

    def _lr(self, niter, lr_base):
        """Calculate learning rate for given base lr and iteration number"""
        return lr_base / (1.0 + niter / self.lr_decay_iterations)

    def get_lr(self):
        return [self._lr(self.last_epoch, base_lr) for base_lr in self.base_lrs]


class CosineFollowedByFlatLR(torch.optim.lr_scheduler._LRScheduler):
    """Pytorch learning rate scheduler.
    Learning rate decreases from the base learning rate
    (set in the optimiser) to lr_min following a
    single half-period of a cosine over lr_cosine_iters iterations.
    The learning rate stays constant at lr_min after lr_cosine_iters
    iterations.
    
    Based on
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """
    def __init__(self, optimizer, lr_min, lr_cosine_iters, last_epoch=-1):
        self.lr_min = lr_min
        self.lr_cosine_iters = lr_cosine_iters
        super(CosineFollowedByFlatLR, self).__init__(optimizer, last_epoch)

    def _lr(self, niter, lr_base):
        """Calculate learning rate for given base lr and iteration number"""
        if niter < self.lr_cosine_iters:
            cos = math.cos(math.pi * niter / self.lr_cosine_iters)
            return self.lr_min + 0.5 * (lr_base - self.lr_min) * (cos + 1.0)
        else:
            return self.lr_min

    def get_lr(self):
        return [self._lr(self.last_epoch, base_lr) for base_lr in self.base_lrs]