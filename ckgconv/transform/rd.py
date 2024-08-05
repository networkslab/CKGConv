# ------- Resistance Distance: Naive implementation --------
from typing import Union, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_scatter import scatter, scatter_add, scatter_max

from torch_geometric.graphgym.config import cfg

from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
)
import torch_sparse
from torch_sparse import SparseTensor
from einops import rearrange, reduce, repeat, einsum

import networkx as nx


def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data
#


@torch.no_grad()
def add_rd(data,
                      walk_length=8,
                      attr_name_abs="spd", # name: 'rrwp'
                      attr_name_rel="spd", # name: ('rrwp_idx', 'rrwp_val')
                      add_identity=True,
                      add_uniform=False,
                      denormalize=False,
                      spd=False,
                      topk: Optional[int]=None,
                      use_sym=False,
                      **kwargs
                      ):

    device=data.edge_index.device
    ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes),
                                       )


    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    # deg_inv = 1.0 / adj.sum(dim=1)
    # deg_inv[deg_inv == float('inf')] = 0
    # adj = adj * deg_inv.view(-1, 1)
    # adj = adj.to_dense()

    adj = adj.to_dense()


    # compute resistance distance via NetworkX
    NxG = pyg.utils.to_networkx(data=data, to_undirected=True, remove_self_loops=True)
    rd = nx.resistance_distance(NxG, nodeA=None, nodeB=None)


    # lazy and stupid way to convert dict from networkx to edge_index in PyG
    row = []
    col = []
    val = []

    for k_1 in rd.keys():
        v_1 = rd[k_1]
        for k_2 in v_1.keys():
            v_2 = v_1[k_2]
            row.append(k_1)
            col.append(k_2)
            val.append(v_2)

    rel_pe_row, rel_pe_col, rel_pe_val = torch.LongTensor(row), torch.LongTensor(col), torch.Tensor(val)



    # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=-2)
    # Fixme: fatal bug --> pyg is right matmul, need row-sum to one, now is col-sum to one.
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)

    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data

