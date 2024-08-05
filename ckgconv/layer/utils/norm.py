


import torch
from torch import nn


class RMSLayerNorm(nn.Module):
    def __init__(self, dim=64, affine=True, bias=True, eps=1e-6):
        super().__init__()

        self.affine = affine
        self.bias = bias

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)

    def forward(self, x):
        std = torch.sqrt(1e-6 + torch.mean(x**2, dim=-1, keepdim=True))
        return x/std * self.gamma + self.beta