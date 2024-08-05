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
def add_full_rrwp_new(data,
                      walk_length=8,
                      attr_name_abs="rrwp", # name: 'rrwp'
                      attr_name_rel="rrwp", # name: ('rrwp_idx', 'rrwp_val')
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
    if use_sym:
        deg = adj.sum(dim=1)
        # adj = adj.to_dense()
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv_sqrt = torch.sqrt(deg_inv)
        adj = adj * deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(1, -1)
        adj = adj.to_dense()
    else:
        deg = adj.sum(dim=1)
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)
        adj = adj.to_dense()

    pe_list = []
    i = 0
    pe_list.append(torch.eye(num_nodes, dtype=torch.float))

    out = adj
    pe_list.append(adj)
    i = i + 1


    if walk_length > 2:
        for j in range(i + 1, walk_length+1):
            out = out @ adj
            pe_list.append(out)


    pe = torch.stack(pe_list, dim=-1) # n x n x k
    abs_pe = pe.diagonal().transpose(0, 1)[:, 1:] # n x k

    if add_uniform:
        pe = torch.cat([pe, (torch.ones_like(adj)/num_nodes).unsqueeze(-1)], dim=-1) # add uniform term (1/N)


    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    ## >>>>>> rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=-2)
    ##### pyg is right matmul, need row-sum to one, now is col-sum to one.
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)

    local_topk = cfg.posenc_RRWP.get('local_topK', None)
    if local_topk is not None and local_topk >= -1:
        local_topk += 0
        mask = rel_pe_val[:, :local_topk+1].sum(dim=-1) > 0
        rel_pe_val = rel_pe_val[mask]
        rel_pe_idx = rel_pe_idx[:, mask]



    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)


    return data

