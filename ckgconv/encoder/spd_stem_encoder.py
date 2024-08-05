'''
    Stem Encoder for Shortest-path Distance:
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



# Note: -------------- Edge/Node-Pair Encoder ----------

@register_edge_encoder('spd_stem')
class SPDStemEdgeEncoder(torch.nn.Module):
    '''
        Relative Stem for RRWP
    '''
    def __init__(self, out_dim, use_bias=True,
                 pad_to_full_graph=True,
                 fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False,
                 pe_name='spd',
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

        self.pe_fc = nn.Linear(emb_dim, out_dim, bias=False)
        self.pe_emb = nn.Embedding(emb_dim, out_dim)
        # trunc_init_(self.pe_fc)
        # nn.init.xavier_normal_(self.pe_fc.weight)


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
            spd_idx = batch[f'{self.pe_name}_index']
            spd_val = batch[f'{self.pe_name}_val']
            out_idx, out_val = spd_idx, spd_val
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

        out_val = self.pe_emb(out_val)
        # out_val = self.pe_fc(self.affine(self.raw_norm(out_val)))

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



#
#
#
#


@torch.no_grad()
def get_num_nodes_per_graph(batch):
    if 'ptr' in batch:
        ptr = batch['ptr']
        return ptr[1:] - ptr[:-1]
    elif 'batch' in batch:
        return scatter(torch.ones_like(batch.batch), index=batch.batch, dim=0, dim_size=batch.num_graphs, reduce='sum')
    else:
        return batch.num_nodes




















class KernelGenerator(nn.Module):
    def __init__(self, cfg=CN()):

        super().__init__()


        self.average =  cfg.attn.get('average', False)
        act = cfg.attn.get('act', 'relu')
        num_blocks = cfg.attn.get('n_mlp_blocks', 1.)
        ffn_ratio = cfg.attn.get('ffn_ratio', 1.)


        if act is None:
            act = nn.Identity
        else:
            act = act_dict[act]

        dim_hidden  = cfg.get('dim_hidden', 64)
        self.num_heads = num_heads = cfg.get('n_heads', dim_hidden)
        out_dim = dim_hidden // num_heads
        hid_dim = out_dim * num_heads

        self.num_groups = num_groups = cfg.get('layers', 10)

        blocks = []
        use_bias = True
        AffineTransformLayer = nn.Identity # Fixme: Temporal Attempts
        for i in range(num_blocks):
            in_fc = GroupedLinear(num_groups, hid_dim, int(hid_dim * ffn_ratio), bias=use_bias)
            out_fc = GroupedLinear(num_groups, int(hid_dim * ffn_ratio),  hid_dim, bias=use_bias)
            nn.init.kaiming_normal_(in_fc.weight, nonlinearity='relu')
            nn.init.xavier_normal_(out_fc.weight)
            # num_groups is counted as receptive field in initialization by default; recover it
            in_fc.weight.data = in_fc.weight.data * np.sqrt(num_groups)
            out_fc.weight.data = out_fc.weight.data * np.sqrt(num_groups)
            blocks = blocks + [
                (nn.Identity(), 'x -> h'),
                (in_fc, 'h -> h'),
                (act(), 'h -> h'),
                (out_fc, 'h -> h'),
                (ResidualLayer(rezero=False, layerscale=False, dim=hid_dim),
                 'h, x -> x'),
            ]

        # todo: attempting strictly following Transformer FFN designs
        fc = GroupedLinear(num_groups, hid_dim, num_heads, bias=use_bias)
        nn.init.xavier_normal_(fc.weight)
        fc.weight.data = fc.weight.data * np.sqrt(num_groups)
        blocks = blocks + [
            (act(), 'x -> x'),
            (fc, 'x -> x')
        ]

        self.blocks = Sequential('x', blocks)
        self.deg_scaler = cfg.attn.get('deg_scaler', True)
        if self.deg_scaler:
            weight = torch.ones(2, 1, num_groups, hid_dim)
            nn.init.xavier_normal_(weight)
            weight = weight.permute(0, 2, 1, 3)
            self.deg_coef = nn.Parameter(weight * np.sqrt(num_groups), requires_grad=True)
            # recover the num_groups which is treated as the receptive field
            # trunc_normal_(self.deg_coef, std=0.02)

        self.register_buffer('replicate_vec', torch.ones(num_groups, 1, 1))

    def forward(self, batch):
        conv_index = batch.conv_index =  batch.edge_index
        conv_loc = batch.edge_attr
        conv_loc = conv_loc.unsqueeze(0)
        index = conv_index[1]

        if self.deg_scaler:
            sqrt_deg = get_sqrt_deg(batch)[index]

            conv_loc = conv_loc * self.deg_coef[0] +\
                       conv_loc * sqrt_deg.unsqueeze(0) * self.deg_coef[1]
        else:
            conv_loc = self.replicate_vec * conv_loc


        kernel = self.blocks(conv_loc)
        score = kernel.view(self.num_groups, -1, self.num_heads, 1)

        batch.conv_kernel = score
        batch.conv_index = conv_index

        return batch























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
