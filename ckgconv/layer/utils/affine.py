

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter





class AffineTransformLayer(nn.Module):
    '''
        Affine Transform Layer;  ResMLP (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9888004)
    '''
    def __init__(self, dim, decay_factor=1.):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim) * decay_factor, requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.init_decay = decay_factor


    def forward(self, x):
        return x * self.gamma + self.beta

    def __repr__(self):
        return f'{super().__repr__()}(dim={self.gamma.size(1)}, init_decay={self.init_decay})'





class RecenterAffineTransformLayer(nn.Module):
    '''
        Recenter to Mean-Ones then Affine Transform
    '''
    def __init__(self, dim, decay_factor=1.):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim) * decay_factor, requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.init_decay = decay_factor

    def forward(self, x, index=None):
        if index is not None:
            avg = scatter(x, index, dim=0, reduce='mean')[index]
        else:
            avg = torch.mean(x, dim=0, keepdims=True)



        return (x - avg + 1) * self.gamma + self.beta

    def __repr__(self):
        return f'{super().__repr__()}(dim={self.gamma.size(1)}, init_decay={self.init_decay})'
