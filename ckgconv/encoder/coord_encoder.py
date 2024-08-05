'''
    The RRWP encoder for GRIT (ours)
'''
import torch
from torch import nn
from torch.nn import functional as F
from ogb.utils.features import get_bond_feature_dims
import torch_sparse

import torch_geometric as pyg
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)

from torch_geometric.utils import remove_self_loops, add_remaining_self_loops
from torch_scatter import scatter
import warnings




@register_node_encoder('coord_linear')
class CoordLinearEncoder(torch.nn.Module):
    '''
        Preprocess the coord
    '''
    def __init__(self, emb_dim, out_dim, attr_name='pe'):
        super().__init__()

        self.encoder = torch.nn.Linear(emb_dim, out_dim)
        self.name = attr_name

    def forward(self, batch):
        x = batch[self.name]
        if x.dim() > 2:
            x = x.transpose(1, -1)

        x = self.encoder(x)

        if x.dim() > 2:
            x = x.transpose(1, -1)

        batch[self.name] = x
        return batch
