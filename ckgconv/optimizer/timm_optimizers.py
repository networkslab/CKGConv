import logging
import math
from typing import Iterator
from dataclasses import dataclass

import torch.optim as optim
from torch.nn import Parameter
from torch.optim import Adagrad, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.graphgym.optim import SchedulerConfig
import torch_geometric.graphgym.register as register

from timm.scheduler.cosine_lr import CosineLRScheduler

@register.register_scheduler('timm_cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int,
                                 min_lr:float=0.,
                                 num_cycles=1.,
                                 reduce_factor=1.,
                                 warmup_lr_init=1e-7,
                                 ):

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=(max_epoch-num_warmup_epochs),
        warmup_t=num_warmup_epochs,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr_init,
        warmup_prefix=True,
        cycle_limit=num_cycles,
        # cycle_decay=reduce_factor,
    )

    if not hasattr(scheduler, 'get_last_lr'):
        # ReduceLROnPlateau doesn't have `get_last_lr` method as of current
        # pytorch1.10; we add it here for consistency with other schedulers.
        def get_last_lr(self):
            """ Return last computed learning rate by current scheduler.
            """
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            return self._last_lr

        scheduler.get_last_lr = get_last_lr.__get__(scheduler)

    def modified_state_dict(ref):
        """Returns the state of the scheduler as a :class:`dict`.
        Additionally modified to ignore 'get_last_lr', 'state_dict'.
        Including these entries in the state dict would cause issues when
        loading a partially trained / pretrained model from a checkpoint.
        """
        return {key: value for key, value in ref.__dict__.items()
                if key not in ['sparsifier', 'get_last_lr', 'state_dict']}

    scheduler.state_dict = modified_state_dict.__get__(scheduler)


    return scheduler

