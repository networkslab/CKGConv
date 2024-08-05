'''
    The RRWP Stem Encoder:
    - modified version for the RRWP-Edge --> v2
    - only differs when using non-global kernels with kernel-rescaling
'''
import torch
import numpy as np
import torch_geometric as pyg
import torch_sparse
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
    act_dict
)
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import Sequential
from torch_scatter import scatter
from yacs.config import CfgNode as CN
import warnings


from ..layer.utils.residual import ResidualLayer
from ..layer.utils.affine import AffineTransformLayer
from ..layer.utils.init import trunc_init_, trunc_normal_


def full_edge_index(edge_index, batch=None):
    """
    Retunr the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full


# Note: -------------- RWSE Encoder ----------

@register_node_encoder('rwse_stem')
class RWSEStemNodeEncoder(torch.nn.Module):
    """
        Node Encoder for RRWP
    """
    def __init__(self, out_dim, use_bias=False, pe_name="rwse"):
        super().__init__()

        kernel_param = cfg.posenc_RWSE.kernel
        emb_dim = len(kernel_param.times)


        if 'stem' not in cfg.gt: cfg.gt.stem = CN()
        self.batch_norm = cfg.gt.stem.get('batch_norm', False)
        self.layer_norm = cfg.gt.stem.get('layer_norm', False)
        self.use_bias = cfg.gt.stem.get('use_bias', False) # Bias Term on FC(RRWP)

        self.sep_norm = cfg.gt.stem.get('sep_norm', False)

        self.use_raw_norm = cfg.gt.stem.get('raw_norm', False)
        self.raw_norm_type = cfg.gt.stem.get('raw_norm_type', 'batch_norm')
        self.use_post_norm = cfg.gt.stem.get('post_norm', True)
        self.post_norm_type = cfg.gt.stem.get('post_norm_type', 'batch_norm')
        if self.use_raw_norm:
            self.raw_norm = nn.BatchNorm1d(emb_dim)
        else:
            self.raw_norm = nn.Identity()

        if self.use_post_norm:
            assert self.post_norm_type == 'batch_norm', '(post-norm) only support batchnorm for now'
            self.post_norm = nn.BatchNorm1d(out_dim)
        else:
            self.post_norm = nn.Identity()

        norm_fn = nn.Identity
        if self.layer_norm: norm_fn = nn.LayerNorm
        if self.batch_norm: norm_fn = nn.BatchNorm1d

        act = cfg.gt.stem.get("act", "relu")
        act = act_dict[act]

        self.name = pe_name
        self.num_layers = cfg.gt.stem.get('num_layers', 0)

        self.pe_fc = nn.Linear(emb_dim, out_dim, bias=self.use_bias)
        trunc_init_(self.pe_fc.weight)


        stem = []
        for i in range(self.num_layers):
            stem += [(act(), 'x -> h'),
                     (nn.Linear(out_dim, out_dim), 'h -> h'),
                     (ResidualLayer(), 'h, x -> x'),
                     (norm_fn(out_dim), 'x -> x')]

        if len(stem) > 0:
            self.stem = pyg.nn.Sequential('x', stem)
            self.stem.apply(trunc_init_)
        else:
            self.stem = nn.Identity()

    def forward(self, batch):
        # Encode just the first dimension if more exist
        rrwp = batch[f"{self.name}"]
        rrwp = self.pe_fc(self.raw_norm(rrwp))
        if self.sep_norm:
            rrwp = self.norm_pe(rrwp)

        if "x" in batch:
            x = batch.x + rrwp
            if self.sep_norm:
                x = self.norm_attr(x)
        else:
            x = rrwp

        batch.x = self.stem(self.post_norm(x))

        return batch


# Note: -------------- Edge/Node-Pair Encoder ----------

@register_edge_encoder('pair_rwse_stem')
class RRWPStemV2EdgeEncoder(torch.nn.Module):
    '''
        Relative Stem for RRWP
    '''
    def __init__(self, out_dim, use_bias=True,
                 pad_to_full_graph=True,
                 fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False,
                 pe_name='rwse',
                 **kwargs
                 ):
        super().__init__()
        # emb_dim = cfg.posenc_RRWP.ksteps + 1
        kernel_param = cfg.posenc_RWSE.kernel
        emb_dim = len(kernel_param.times)

        self.pe_name = pe_name
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop

        if 'stem' not in cfg.gt: cfg.gt.stem = CN()
        self.batch_norm = cfg.gt.stem.get('batch_norm', False)
        self.layer_norm = cfg.gt.stem.get('layer_norm', False)
        self.use_bias = cfg.gt.stem.get('use_bias', False) # Bias Term on FC(RRWP)
        self.add_scaled_bias = cfg.gt.stem.get('add_scaled_bias', False)
        self.scale_attr = cfg.gt.stem.get('scale_attr', False)
        self.concat_attr = cfg.gt.stem.get('concat_attr', False) # will extend the dimension of edges
        if self.add_scaled_bias:
            emb_dim += 1

        self.use_raw_norm = cfg.gt.stem.get('raw_norm', True)
        self.use_post_norm = cfg.gt.stem.get('post_norm', True)
        self.raw_norm_type = cfg.gt.stem.get('raw_norm_type', 'batch_norm')
        self.post_norm_type = cfg.gt.stem.get('post_norm_type', 'batch_norm')
        self.affine_transform = cfg.gt.stem.get('affine_transform', True)
        if self.use_raw_norm:
            assert self.raw_norm_type == 'batch_norm', '(raw-norm) only support batchnorm for now'
            affine = not self.affine_transform
            self.raw_norm = nn.BatchNorm1d(emb_dim, affine=affine)
        else:
            self.raw_norm = nn.Identity()

        self.affine_alpha = cfg.gt.stem.get('affine_alpha', 0.1)
        if self.affine_transform:
            self.affine = AffineTransformLayer(emb_dim, decay_factor=self.affine_alpha)
        else:
            self.affine = nn.Identity()

        if self.use_post_norm:
            assert self.post_norm_type == 'batch_norm', '(post-norm) only support batchnorm for now'
            self.post_norm = nn.BatchNorm1d(out_dim)
        else:
            self.post_norm = nn.Identity()

        # compute the expected value per kernel instead of the raw-RRWP


        kernel_expected = cfg.gt.stem.get('kernel_expected', False)
        self.kernel_rescale = cfg.gt.stem.get('kernel_rescale', False)
        if kernel_expected:
            self.kernel_rescale = True

        self.k_hop = cfg.gt.stem.get('k_hop', -1)
        self.no_edge_attr = cfg.gt.stem.get('no_edge_attr', False)

        self.rwse_src_fc = nn.Linear(emb_dim, out_dim, bias=False)
        self.rwse_dst_fc = nn.Linear(emb_dim, out_dim, bias=False)
        # trunc_init_(self.pe_fc)
        nn.init.xavier_normal_(self.rwse_src_fc.weight)
        nn.init.xavier_normal_(self.rwse_dst_fc.weight)


        self.use_loc = False
        if cfg.dataset.name in ['CIFAR10', 'MNIST']:
            self.use_loc = True
            self.loc_fc_src = nn.Linear(2, out_dim, bias=False)
            self.loc_fc_dst = nn.Linear(2, out_dim, bias=False)
            nn.init.xavier_normal_(self.loc_fc_src.weight)
            nn.init.xavier_normal_(self.loc_fc_dst.weight)


        self.pad_to_full_graph = cfg.gt.attn.get('full_attn', False)
        self.fill_value = 0.

        if self.batch_norm:
            norm_fn = nn.BatchNorm1d
        elif self.layer_norm:
            norm_fn = nn.LayerNorm
        else:
            norm_fn = nn.Identity

        self.num_layers = cfg.gt.stem.get('num_layers', 0)
        act = cfg.gnn.get("act", "relu")
        act = act_dict[act]

        ffn_ratio = cfg.gt.stem.get("ffn_ratio", 1)

        stem = []
        for i in range(self.num_layers):
            in_fc = nn.Linear(out_dim, int(out_dim * ffn_ratio))
            out_fc = nn.Linear(int(out_dim * ffn_ratio), out_dim)
            stem += [
                     (in_fc, 'x -> h'),
                     (act(), 'h -> h'),
                     (out_fc, 'h -> h'),
                     (ResidualLayer(), 'h, x -> x'),
                     (norm_fn(out_dim), 'x -> x')]

        if len(stem) > 0:
            self.stem = pyg.nn.Sequential('x', stem)
            self.stem.apply(trunc_init_)
        else:
            self.stem = nn.Identity()

    def forward(self, batch):

        nodes_per_graph = batch.ptr[1:] - batch.ptr[:-1]
        # TODO: only support mini-batched graphs for now

        with torch.no_grad():
            out_idx = full_edge_index(batch.edge_index, batch=batch.batch)
            # zero padding to fully-connected graphs

        out_idx, _ = add_self_loops(out_idx, None, num_nodes=batch.num_nodes, fill_value=0.)
        out_idx, _ = pyg.utils.coalesce(out_idx, None, batch.num_nodes, reduce='add')

        rwse = batch.rwse
        out_val = self.rwse_src_fc(rwse)[out_idx[0]] + self.rwse_dst_fc(rwse)[out_idx[1]]

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        batch.raw_edge_index = batch.edge_index
        batch.raw_edge_attr = batch.edge_attr

        if not self.no_edge_attr:
            if edge_attr is None:
                edge_attr = edge_index.new_zeros(edge_index.size(1), out_val.size(1))

            out_idx, out_val = torch.cat([edge_index, out_idx], dim=1), \
                torch.cat([edge_attr, out_val], dim=0)

            out_idx, out_val = torch_sparse.coalesce(
                out_idx, out_val, batch.num_nodes, batch.num_nodes,
                op="add"
            )

        if 'pos' in batch:
            loc = self.loc_fc_src(batch.pos[out_idx[0]]) + self.loc_fc_dst(batch.pos[out_idx[1]])
            out_val = out_val + loc

        out_val = self.post_norm(out_val)

        out_val = self.stem(out_val)
        batch.edge_index, batch.edge_attr = out_idx, out_val

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"{super().__repr__()}" \
               f"[pad_to_full_graph={self.pad_to_full_graph}]" \




@torch.no_grad()
def get_num_nodes_per_graph(batch):
    if 'ptr' in batch:
        ptr = batch['ptr']
        return ptr[1:] - ptr[:-1]
    elif 'batch' in batch:
        return scatter(torch.ones_like(batch.batch), index=batch.batch, dim=0, dim_size=batch.num_graphs, reduce='sum')
    else:
        return batch.num_nodes














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
