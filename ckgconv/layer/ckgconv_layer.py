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

from .utils.residual import ResidualLayer
from .utils.affine import AffineTransformLayer, RecenterAffineTransformLayer
from .utils.init import trunc_init_



def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out



    def __repr__(self):
        return f'{super().__repr__()}(rezero={self.rezero}, layerscale={self.layerscale}, layerscale_init={self.layerscale_init}, dim={self.dim})'


class GraphSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, index, num_nodes=None):
        return pyg_softmax(src, index, num_nodes=num_nodes)



class CKGConv(nn.Module):
    """
        CKGConv
    """

    def __init__(self, in_dim, out_dim, num_heads,
                 kernel_size=-1, # -1 stands for global filter,
                 dilation=1,
                 use_bias=True,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 omega=30,
                 cfg=CN(),
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        # cfg here is CFG.gt
        self.batch_norm = cfg.attn.get('batch_norm', False)
        self.weight_norm = cfg.attn.get('weight_norm', False)
        self.layer_norm = cfg.attn.get('layer_norm', False)
        self.graph_norm = cfg.attn.get('graph_norm', False)
        self.kernel_norm = cfg.attn.get('kernel_norm', False)
        self.group_norm = cfg.attn.get('group_norm', False)
        self.deg_scaler = cfg.attn.get('deg_scaler', False)


        self.ffn_ratio = cfg.attn.get('ffn_ratio', 1.)
        self.average =  cfg.attn.get('average', True)
        self.dynamic_avg = cfg.attn.get('dynamic_avg', False)
        self.blur_kernel = cfg.attn.get('blur_kernel', False)

        if act is None:
            act = nn.Identity
        else:
            act = act_dict[act]

        num_blocks = cfg.attn.get('n_mlp_blocks', 1)

        # norm_args = 'x -> x'
        # norm_fn = nn.Identity
        if self.batch_norm:
            norm_fn = nn.BatchNorm1d
        elif self.layer_norm:
            norm_fn = nn.LayerNorm
        else:
            # by default adding one affine Transformation for gelu
            norm_fn = AffineTransformLayer

        self.mlp_dropout = cfg.attn.get('mlp_dropout', 0.)
        hid_dim = out_dim * num_heads

        use_bias = True
        blocks = []


        if cfg.attn.get('ffn', False): # FFN-like MLP
            for i in range(num_blocks):
                blocks = blocks + [
                    (nn.Linear(hid_dim, int(hid_dim * self.ffn_ratio), bias=use_bias), 'x -> h'),
                    (norm_fn(int(hid_dim * self.ffn_ratio)), 'h -> h'),
                    (act(), 'h -> h'),
                    (nn.Linear(int(hid_dim * self.ffn_ratio),  hid_dim, bias=use_bias), 'h -> h'),
                    (nn.Dropout(self.mlp_dropout) if self.mlp_dropout > 0 else nn.Identity(), 'h -> h') ,
                    (ResidualLayer(rezero=False, layerscale=False, dim=hid_dim),'h, x -> x'),
                ]
            blocks = blocks + [
                (nn.Linear(hid_dim, hid_dim, bias=use_bias), 'x -> x')
            ]
        else: # ResNetV2-like MLP
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
                (nn.Linear(hid_dim, hid_dim, bias=use_bias), 'x -> x')
            ]

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.blocks = Sequential('x, index', blocks)
        self.blocks.apply(trunc_init_)

        if self.weight_norm:
            self.blocks.apply(_wn_linear)

        self.value_proj = cfg.attn.get('value_proj', False)
        if self.value_proj:
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.bias = nn.Parameter(torch.zeros(1, num_heads, out_dim), requires_grad=True)
            self.V.apply(trunc_init_)
        else:
            self.V = nn.Identity()
            self.bias = nn.Parameter(torch.zeros(1, num_heads, out_dim), requires_grad=False)

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, hid_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)


    def propagate_attention(self, batch):

        edge_index = batch.edge_index
        E = batch.edge_attr

        E = self.blocks(E, None)
        score = E.view(-1, self.num_heads, 1)


        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        if CFG.train.mode == 'log-conv-kernel':
            self.kernel_val = score
            self.kernel_index = edge_index

        if self.blur_kernel:
            score = pyg_softmax(score, edge_index[1])
            score = self.attn_dropout(score)
            msg = batch.V_h[edge_index[0]] * score  # (num relative) x num_heads x out_dim
            batch.wV = scatter(msg, edge_index[1],
                               dim=0, dim_size=batch.num_nodes,
                               reduce='add')
        else:
            score = self.attn_dropout(score)
            msg = batch.V_h[edge_index[0]] * score  # (num relative) x num_heads x out_dim
            batch.wV = scatter(msg, edge_index[1],
                               dim=0, dim_size=batch.num_nodes,
                               reduce='mean')

        if self.deg_scaler:
            sqrt_deg = get_sqrt_deg(batch)
            h = batch.wV.view(batch.wV.size(0), -1)
            h = h * self.deg_coef[:,:, 0] + h * sqrt_deg * self.deg_coef[:, :, 1]
            batch.wV = h.view(-1, self.num_heads, self.out_dim)

        batch.wV = batch.wV + self.bias

    def forward(self, batch):
        V_h = self.V(batch.x)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV

        return h_out


@register_layer("CKGraphConvMLP")
class CKGraphConvLayer(nn.Module):
    """
        CKGConv-Block: input -> CKGConv -> FFN -> output
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 kernel_size=-1, # determined by shortest-path distance
                 dilation=1,   # determines by shortest-path distance
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.use_drop_path = cfg.get('use_drop_path', False)
        # use drop_path (drop a sample)
        # -------
        self.act = act_dict[act]() if act is not None else nn.Identity()

        self.conv = CKGConv(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            # kernel_size=kernel_size,
            # dilation=dilation,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", None),
            act=cfg.attn.get("act", "gelu"),
            cfg=cfg
        )

        self.out_proj = cfg.attn.get('out_proj', True)
        if self.out_proj:
            self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
            nn.init.xavier_normal_(self.O_h.weight)
        else:
            self.O_h = nn.Identity()

        # -------- Deg Scaler Option ------
        norm_fn = nn.Identity
        if self.layer_norm: norm_fn = nn.LayerNorm
        if self.batch_norm: norm_fn = partial(nn.BatchNorm1d)

        ffn_ratio = cfg.get('ffn_ratio', 2)
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * int(ffn_ratio))
        self.FFN_h_layer2 = nn.Linear(out_dim * int(ffn_ratio), out_dim)
        self.FFN_h_layer1.apply(trunc_init_)
        self.FFN_h_layer2.apply(trunc_init_)

        self.norm1_h  = norm_fn(out_dim)
        self.norm2_h = norm_fn(out_dim)

        self.dropout = nn.Dropout(self.dropout) if dropout > 0 else nn.Identity()

        # residual connection
        self.res_layer1 = ResidualLayer(rezero=False, layerscale=False, dim=out_dim)
        self.res_layer2 = ResidualLayer(rezero=False, layerscale=False, dim=out_dim)



    def _norm_layer(self, norm_fn,  h, batch_index=None):
        h = norm_fn(h)

        return h


    def forward(self, batch):
        num_nodes = batch.num_nodes
        h = h_in1 = batch.x # for first residual connection

        h_conv_out = self.conv(batch)
        h = h_conv_out.view(num_nodes, -1)
        h = self.O_h(h)

        h = self.dropout(h)
        h = self.res_layer1(h, h_in1)
        h = self._norm_layer(self.norm1_h, h, batch_index=batch.batch)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = self.FFN_h_layer2(h)

        h = self.dropout(h)
        h = self.res_layer2(h, h_in2)
        h = self._norm_layer(self.norm2_h, h, batch_index=batch.batch)

        batch.x = h

        return batch

    def __repr__(self):
        norm_type = 'post-norm'
        return '{}(in_channels={}, out_channels={}, heads={}, residual={}, norm_type={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
            norm_type,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float
                               )
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg

@torch.no_grad()
def get_sqrt_deg(batch):
    if "deg" in batch:
        deg = batch.deg
        sqrt_deg = torch.sqrt(deg).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.raw_edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float
                               )
        sqrt_deg = torch.sqrt(deg)
    sqrt_deg = sqrt_deg.view(batch.num_nodes, 1)
    return sqrt_deg


def _init_linear(l):
    if isinstance(l, nn.Linear): nn.init.xavier_normal_(l.weight)

def _wn_linear(l):
    if isinstance(l, nn.Linear):
        nn.utils.weight_norm(l, name='weight')

def xavier_normal_(l):
    if isinstance(l, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_normal_(l.weight)
    elif isinstance(l, nn.Parameter):
        nn.init.xavier_normal_(l)

def signed_sqrt(x):
    return torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))