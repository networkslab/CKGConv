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

from einops import repeat

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


def forward(batch):
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    out_idx, out_val = edge_index, edge_attr

    edge_index_full = full_edge_index(out_idx, batch=batch.batch)
    # edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
    edge_attr_pad = edge_attr.new_zeros(edge_index_full.size(1), edge_attr.size(1))
    # zero padding to fully-connected graphs
    out_idx = torch.cat([out_idx, edge_index_full], dim=1)
    out_val = torch.cat([out_val, edge_attr_pad], dim=0)
    out_idx, out_val = torch_sparse.coalesce(
        out_idx, out_val, batch.num_nodes, batch.num_nodes,
        op="add"
    )

    batch.edge_index, batch.edge_attr = out_idx, out_val
    return batch


@register_edge_encoder('pad_to_full_graph')
class PadToFullGraphEdgeEncoder(torch.nn.Module):
    '''
        Padding to Full Attention
    '''
    def __init__(self, out_dim, add_identity=True, **kwargs):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.out_dim = out_dim
        self.pad_to_full_graph = True
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_bias = nn.Parameter(torch.zeros(1, out_dim))
            nn.init.xavier_normal_(self.identity_bias)

    def forward(self, batch):
        edge_index = batch.edge_index
        if 'edge_attr' not in batch:
            edge_attr = edge_index.new_zeros(edge_index.size(1), self.out_dim)
        else:
            edge_attr = batch.edge_attr

        edge_index_full = full_edge_index(edge_index, batch=batch.batch)
        edge_attr_pad = edge_attr.new_zeros(edge_index_full.size(1), edge_attr.size(1))
        edge_index = torch.cat([edge_index, edge_index_full], dim=-1)
        edge_attr = torch.cat([edge_attr, edge_attr_pad], dim=0)

        if self.add_identity:
            edge_index_sl = repeat(torch.arange(batch.num_nodes, device=edge_attr.device), 'n -> b n', b=2)
            edge_attr_sl = repeat(self.identity_bias, 'i d -> (b i) d', b=batch.num_nodes)
            edge_index = torch.cat([edge_index, edge_index_sl], dim=-1)
            edge_attr = torch.cat([edge_attr, edge_attr_sl], dim=0)

        edge_index, edge_attr = pyg.utils.coalesce(edge_index, edge_attr=edge_attr, num_nodes=batch.num_nodes)

        batch.edge_index = edge_index
        batch.edge_attr = edge_attr

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"

#
@register_edge_encoder('add_full_graph_index')
class AddFullGraphEdgeEncoder(torch.nn.Module):
    '''
        Add full-graph edge-index
    '''
    def __init__(self, emb_dim=None, add_identity=True, **kwargs):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.add_identity = add_identity

    def forward(self, batch):
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        edge_index = full_edge_index(edge_index, batch=batch.batch)
        if self.add_identity:
            edge_index = pyg.utils.add_self_loops(edge_index, num_nodes=batch.num_nodes)[0]

        edge_index = pyg.utils.coalesce(edge_index, num_nodes=batch.num_nodes)
        batch.full_edge_index = edge_index

        return batch


@register_edge_encoder('add_self_identity')
class AddFullGraphEdgeEncoder(torch.nn.Module):
    '''
        Add full-graph edge-index
    '''
    def __init__(self, out_dim=None, add_identity=True, **kwargs):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_normal_(self.bias)
        self.out_dim = out_dim

    def forward(self, batch):
        edge_index = batch.edge_index
        edge_attr = batch.get('edge_attr')
        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), self.out_dim)

        edge_index, edge_attr = pyg.utils.add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=batch.num_nodes, fill_value=self.bias)

        edge_index, edge_attr = pyg.utils.coalesce(edge_index, edge_attr=edge_attr, num_nodes=batch.num_nodes)
        batch.full_edge_index = edge_index

        return batch
