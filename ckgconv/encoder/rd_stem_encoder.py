'''
    Stem Encoder for Resistance Distance (RD)
    - Used Gaussian Basis Kernels
'''

import torch
from torch import nn
from torch.nn import functional as F
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




# Note: -------------- Edge/Node-Pair Encoder ----------

@register_edge_encoder('rd_stem')
class RDStemEdgeEncoder(torch.nn.Module):
    '''
        Relative Stem for RRWP
    '''
    def __init__(self, out_dim, use_bias=True,
                 pad_to_full_graph=True,
                 fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False,
                 pe_name='rd',
                 **kwargs
                 ):
        super().__init__()
        emb_dim = cfg.posenc_RRWP.ksteps + 1
        if cfg.posenc_RRWP.add_uniform:
            emb_dim = emb_dim + 1

        if cfg.posenc_RRWP.add_attr:
            emb_dim = cfg.posenc_RRWP.ksteps * (cfg.dataset.edge_encoder_num_types + 1) + 1

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


        # kernel_expected = cfg.gt.stem.get('kernel_expected', False)
        # self.kernel_rescale = cfg.gt.stem.get('kernel_rescale', False)
        # if kernel_expected:
        #     self.kernel_rescale = True

        self.k_hop = cfg.gt.stem.get('k_hop', -1)
        self.no_edge_attr = cfg.gt.stem.get('no_edge_attr', False)

        self.pe_gbk = GaussianLayer(out_dim)
        # Gaussian Basis Kernel



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
            out_idx = batch[f'{self.pe_name}_index']
            out_val = batch[f'{self.pe_name}_val']
            # if self.k_hop >= 0:
            #     mask = out_val[:, :self.k_hop+1].sum(dim=-1, keepdim=True) > 0
            #     mask = mask.flatten()
            #     out_idx, out_val = out_idx[:, mask], out_val[mask]
            # elif self.pad_to_full_graph:
            #     edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            #     edge_attr_pad = out_val.new_zeros(edge_index_full.size(1), out_val.size(1))
            #     # zero padding to fully-connected graphs
            #     out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            #     out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            #
            # out_idx, out_val = add_self_loops(out_idx, out_val, num_nodes=batch.num_nodes, fill_value=0.)
            # out_idx, out_val = torch_sparse.coalesce(
            #     out_idx, out_val, batch.num_nodes, batch.num_nodes,
            #     op="add"
            # )
            # #

        out_val = self.pe_gbk(out_val)

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
               f"[pad_to_full_graph={self.pad_to_full_graph}]"



class GaussianLayer(nn.Module):
    '''
    Modified from [GD-Graphormer](https://github.com/lsj2408/Graphormer-GD/blob/master/graphormer/modules/graphormer_layers.py#L245)
    - remove the edge-types for simple graphs
    '''
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        # nn.init.constant_(self.bias.weight, 0)
        # nn.init.constant_(self.mul.weight, 1)

    def forward(self, x):
        x = x.view(-1, 1)
        mean = self.means.weight.float()
        std = self.stds.weight.float().abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)




@torch.no_grad()
def get_num_nodes_per_graph(batch):
    if 'ptr' in batch:
        ptr = batch['ptr']
        return ptr[1:] - ptr[:-1]
    elif 'batch' in batch:
        return scatter(torch.ones_like(batch.batch), index=batch.batch, dim=0, dim_size=batch.num_graphs, reduce='sum')
    else:
        return batch.num_nodes



class GraphRDBias(nn.Module):
    """
        Compute 3D attention bias according to the position information for each head.
        """

    def __init__(self, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share=False, no_node_feature=False):
        super(GraphRDBias, self).__init__()
        self.num_heads = num_heads
        self.num_edges = num_edges
        self.n_layers = n_layers
        self.no_share = no_share
        self.num_kernel = num_kernel
        self.embed_dim = embed_dim


        rd_bias_heads = self.num_heads * self.n_layers if self.no_share else self.num_heads
        self.gbf = GaussianLayer(self.num_kernel, num_edges)
        self.gbf_proj = NonLinear(self.num_kernel, rd_bias_heads)

        if self.num_kernel != self.embed_dim:
            self.edge_proj = nn.Linear(self.num_kernel, self.embed_dim)
        else:
            self.edge_proj = None

        self.no_node_feature = no_node_feature

    def forward(self, batched_data):

        res_pos, x, node_type_edge = batched_data['res_pos'], batched_data['x'], batched_data['node_type_edge'] # pos shape: [n_graphs, n_nodes, 3]
        # pos.requires_grad_(True)

        padding_mask = x.eq(0).all(dim=-1)
        n_graph, n_node, _ = res_pos.shape
        dist = res_pos

        edge_feature = self.gbf(dist, torch.zeros_like(node_type_edge).long() if node_type_edge is None or self.no_node_feature else node_type_edge.long())
        gbf_result = self.gbf_proj(edge_feature)
        graph_attn_bias = gbf_result

        graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
        graph_attn_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )

        edge_feature = edge_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0
        )

        sum_edge_features = edge_feature.sum(dim=-2)
        merge_edge_features = self.edge_proj(sum_edge_features)

        merge_edge_features = merge_edge_features.masked_fill(
            padding_mask.unsqueeze(-1).to(torch.bool), 0.0
        )

        return graph_attn_bias, merge_edge_features




class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x