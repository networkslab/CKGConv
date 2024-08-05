'''
    Simplified CKGConv without using cfg file from GraphGym
    > for simple case study.
'''

import os
from os.path import join
save_path = os.getcwd()
os.chdir(join(os.getcwd(), '..'))


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add

from ckgconv.utils import negate_edge_index
from torch_geometric.graphgym.register import *

from yacs.config import CfgNode as CN
import warnings

from torch_geometric.nn import Sequential
from functools import partial

from torch_geometric.graphgym.config import cfg as CFG

from ckgconv.layer.utils.residual import ResidualLayer
from ckgconv.layer.utils.affine import AffineTransformLayer, RecenterAffineTransformLayer
from ckgconv.layer.utils.init import trunc_init_
from torch_geometric.utils import softmax as pyg_softmax



class CKGConv(nn.Module):
    """
        Simplified CKGConv
        > no deg-scaler for now
    """

    def __init__(self, in_dim, out_dim,
                 pe_dim,
                 clamp=None, act=nn.GELU,
                 batch_norm=False,
                 layer_norm=False,
                 # deg_scaler=False,
                 ffn_ratio=1.,
                 average=True,
                 num_blocks=1,
                 mlp_dropout=0.,
                 attn_dropout=0.,
                 out_proj=True,
                 softmax=False,
                 softplus=False,
                 **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pe_dim = pe_dim
        self.clamp = np.abs(clamp) if clamp is not None else None

        # cfg here is CFG.gt
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        # self.deg_scaler = deg_scaler
        self.softmax = softmax
        self.softplus = softplus


        self.ffn_ratio = ffn_ratio
        self.average = average

        num_blocks = num_blocks

        if self.batch_norm:
            norm_fn = nn.BatchNorm1d
        elif self.layer_norm:
            norm_fn = nn.LayerNorm
        else:
            norm_fn = nn.Identity

        self.mlp_dropout = mlp_dropout

        use_bias = True

        blocks = []

        hid_dim = max(in_dim, pe_dim)
        if pe_dim != in_dim:
            blocks += [(nn.Linear(pe_dim, hid_dim), 'x -> x')]


        for i in range(num_blocks):
            blocks = blocks + [
                (norm_fn(hid_dim), 'x -> h'),
                (act(), 'h -> h'),
                (nn.Linear(hid_dim, int(hid_dim * self.ffn_ratio), bias=use_bias), 'h -> h'),
                (norm_fn(int(hid_dim * self.ffn_ratio)), 'h -> h'),
                (act(), 'h -> h'),
                (nn.Linear(int(hid_dim * self.ffn_ratio),  hid_dim, bias=use_bias), 'h -> h'),
                (nn.Dropout(self.mlp_dropout) if self.mlp_dropout > 0 else nn.Identity(), 'h -> h') ,
                (ResidualLayer(rezero=False, layerscale=False, dim=hid_dim),'h, x -> x'),
            ]

        blocks = blocks + [
            (norm_fn(hid_dim), 'x -> x'),
            (nn.Linear(hid_dim, in_dim, bias=use_bias), 'x -> x')
        ]

        if self.softplus:
            blocks += [(nn.Softplus(), 'x -> x')]

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

        self.blocks = Sequential('x, index', blocks)
        # self.blocks.apply(trunc_init_)

        self.out_proj = out_proj
        if self.out_proj:
            self.O = nn.Linear(in_dim, out_dim, bias=True)
            # self.O.apply(trunc_init_)
        else:
            self.O = nn.Identity()


    def propagate_attention(self, x, pe_index, pe_val, deg=None):

        edge_index = pe_index
        E = pe_val

        E = self.blocks(E, None)
        score = E
        reduce = 'mean' if self.average else 'add'

        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp.abs(), max=self.clamp.abs())

        if self.softmax:
            reduce='add'
            score = pyg_softmax(score, edge_index[1], num_nodes=x.size(0), dim=0)


        self.kernel = score
        score = self.attn_dropout(score)
        msg = x[edge_index[0]] * score  # (num relative) x num_heads x out_dim

        wV = scatter(msg, edge_index[1],
                           dim=0, dim_size=x.size(0),
                           reduce=reduce)

        return wV

    def forward(self, x, pe_index, pe_val, deg=None):
        # batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        wV = self.propagate_attention(x, pe_index, pe_val, deg)
        h_out = self.O(wV)

        return h_out
