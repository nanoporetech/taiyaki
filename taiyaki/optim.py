# Code used by training scripts.

import math
import torch


class ReciprocalLR(torch.optim.lr_scheduler._LRScheduler):
    """Pytorch learning rate scheduler.
    Learning rate is the base learning rate
    (set in the optimiser), multiplied by
   
    1 / (1 + niter/lr_decay_iterations),
   
    so that after lr_decay iterations, the l.r. is divided by 2.

    Also allows a warmup period where learning rate is at lr_warmup for warmup_iters
    before all this starts.
   
    Based on
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """
    def __init__(self, optimizer, lr_decay_iterations,
                 warmup_iters=0, lr_warmup=1.0e-16, last_epoch=-1):
        self.lr_decay_iterations = lr_decay_iterations
        self.warmup_iters = warmup_iters
        self.lr_warmup = lr_warmup
        super(ReciprocalLR, self).__init__(optimizer, last_epoch)

    def _lr(self, niter, lr_base):
        """Calculate learning rate for given base lr and iteration number"""
        if niter < self.warmup_iters:
            return self.lr_warmup
        post_warmup_iters = niter - self.warmup_iters
        return lr_base / (1.0 + post_warmup_iters / self.lr_decay_iterations)

    def get_lr(self):
        return [self._lr(self.last_epoch, base_lr) for base_lr in self.base_lrs]


class CosineFollowedByFlatLR(torch.optim.lr_scheduler._LRScheduler):
    """Pytorch learning rate scheduler.
    Learning rate decreases from the base learning rate
    (set in the optimiser) to lr_min following a
    single half-period of a cosine over lr_cosine_iters iterations.
    The learning rate stays constant at lr_min after lr_cosine_iters
    iterations.
   
    Also allows a warmup period where learning rate is at lr_warmup for warmup_iters
    before all this starts.
   
    Based on
    https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
    """
    def __init__(self, optimizer, lr_min, lr_cosine_iters,
                                  warmup_iters=0, lr_warmup=1.0e-16, last_epoch=-1):
        self.lr_min = lr_min
        self.lr_cosine_iters = lr_cosine_iters
        self.warmup_iters = warmup_iters
        self.lr_warmup = lr_warmup
        super(CosineFollowedByFlatLR, self).__init__(optimizer, last_epoch)

    def _lr(self, niter, lr_base):
        """Calculate learning rate for given base lr and iteration number"""
        if niter < self.warmup_iters:
            return self.lr_warmup
        post_warmup_iters = niter - self.warmup_iters
        if post_warmup_iters < self.lr_cosine_iters:
            cos = math.cos(math.pi * post_warmup_iters / self.lr_cosine_iters)
            return self.lr_min + 0.5 * (lr_base - self.lr_min) * (cos + 1.0)
        else:
            return self.lr_min

    def get_lr(self):
        return [self._lr(self.last_epoch, base_lr) for base_lr in self.base_lrs]