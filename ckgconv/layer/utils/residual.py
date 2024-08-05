

import torch
from torch import nn
from torch.nn import functional as F





class ResidualLayer(nn.Module):
    '''
        Residual Layer;
        - support rezero and layerscale
    '''
    def __init__(self, rezero=False, layerscale=False, alpha=0.1, dim=1):
        super().__init__()
        self.rezero = rezero
        self.layerscale = layerscale
        self.layerscale_init = alpha

        if rezero:
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
            # self.alpha = nn.Parameter(torch.ones(1) * 1e-6, requires_grad=True)
            # rezero is set to zero be default
        elif layerscale:
            self.alpha = nn.Parameter(torch.ones(1, dim) * alpha, requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)

        self.dim = self.alpha.size(-1)
    def forward(self, x, x_res):
        if not self.rezero and not self.layerscale:
           return x + x_res

        return x * self.alpha + x_res

    def __repr__(self):
        return f'{super().__repr__()}(rezero={self.rezero}, layerscale={self.layerscale}, ' \
               f'layerscale_init={self.layerscale_init}, ' \
               f'alpha_shape={self.alpha.shape})'