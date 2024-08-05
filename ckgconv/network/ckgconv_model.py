import torch
from torch import nn
import torch_geometric as pyg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from functools import partial




class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner

        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


# class PosencEncoder(torch.nn.Module):
class Stem(torch.nn.Module):
    """
        Encoding node and edge Positional Encoding
        Args:
            dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, **kwargs):
        super().__init__()


        node_stem = cfg.gt.stem.get('node_stem', None)
        if node_stem is None: # to be compatible with previous versions
            node_stem = cfg.gt.get('node_pe_encoder', None)

        if node_stem is None: node_stem=''
        node_stem = node_stem.split("+")
        self.node_stem = nn.Sequential(*[register.node_encoder_dict[enc](out_dim=cfg.gnn.dim_inner,
                                                                           )
                                           for enc in node_stem if enc != ''])

        edge_stem = cfg.gt.stem.get('edge_stem', None)
        if edge_stem is None: # to be compatible with previous versions
            edge_stem = cfg.gt.get('edge_pe_encoder', None)

        if edge_stem is None: edge_stem=''
        edge_stem = edge_stem.split("+")
        self.edge_stem = nn.Sequential(*[register.edge_encoder_dict[enc](out_dim=cfg.gnn.dim_inner,
                                                                          )
                                           for enc in edge_stem if enc != ''])

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('CKGConvNet')
class CKGraphConvNet(torch.nn.Module):
    '''
        The proposed Continuous Kernel Convolution Networks
    '''

    def __init__(self, dim_in, dim_out):
        super().__init__()

        if cfg.gnn.dim_inner == -1:
            cfg.gnn.dim_inner = cfg.gt.dim_hidden

        self.feat_enc = FeatureEncoder(dim_in)
        dim_in = self.feat_enc.dim_in
        # self.pe_encoder = PosencEncoder(dim_in)
        self.stem = Stem(dim_in)

        # pre-backbone normalization
        pre_backbone_norm = cfg.gt.get('pre_backbone_norm', False)
        norm_fn = nn.Identity
        if cfg.gt.batch_norm: norm_fn = partial(nn.BatchNorm1d, momentum=cfg.gt.bn_momentum)
        if cfg.gt.layer_norm: norm_fn = nn.LayerNorm
        graph_norm = False
        if cfg.gt.get('graph_norm', False):
            norm_fn = pyg.nn.GraphNorm
            graph_norm = True

        if pre_backbone_norm:
            self.pre_backbone_norm = NormalizationLayer(norm_fn(cfg.gt.dim_hidden), graph_norm=graph_norm)
        else:
            self.pre_backbone_norm = nn.Identity()

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        global_model_type = cfg.gt.get('layer_type', "CKGraphConvMLP")
        # global_model_type = "GritTransformer"

        ConvBlock = register.layer_dict.get(global_model_type)
        kernel_size = cfg.gt.get('kernel_size', -1)
        dilation = cfg.gt.get('dilation', 1)
        if isinstance(kernel_size, str):
            kernel_size = [int(i) for i in kernel_size.split(',')]
        else:
            kernel_size = [kernel_size] * cfg.gt.layers


        if isinstance(dilation, str):
            dilation = [int(i) for i in dilation.split(',')]
        elif isinstance(dilation, tuple):
            pass
        else:
            dilation = [dilation] * cfg.gt.layers


        layers = []
        for l in range(cfg.gt.layers):
            layers.append(ConvBlock(
                in_dim=cfg.gt.dim_hidden,
                out_dim=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                kernel_size=kernel_size[l],
                dilation=dilation[l],
                dropout=cfg.gt.dropout,
                act=cfg.gnn.act,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=True,
                norm_e=cfg.gt.attn.norm_e,
                O_e=cfg.gt.attn.O_e,
                cfg=cfg.gt,
                out_norm=l==cfg.gt.layers-1
                # log_attn_weights=cfg.train.mode == 'log-attn-weights',
            ))

        # if global_model_type == "Norm-Res-GritTransformer" or global_model_type == "PreNormGritTransformer":
        #     layers.append(register.layer_dict["GeneralNormLayer"]\
        #         (dim=cfg.gt.dim_hidden,
        #          layer_norm=cfg.gt.layer_norm,
        #          batch_norm=cfg.gt.batch_norm,
        #          cfg=cfg.gt
        #          ))

        self.layers = torch.nn.Sequential(*layers)

        # pre-backbone normalization
        post_backbone_norm = cfg.gt.get('post_backbone_norm', False)
        if post_backbone_norm:
            self.post_backbone_norm = NormalizationLayer(norm_fn(cfg.gt.dim_hidden), graph_norm=graph_norm)
        else:
            self.post_backbone_norm = nn.Identity()

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):

        for module in self.children():
            batch = module(batch)

        return batch



class NormalizationLayer(nn.Module):
    def __init__(self, norm_layer, graph_norm=False):
        super().__init__()
        self.norm_layer = norm_layer
        self.graph_norm = graph_norm

    def forward(self, batch):
        if self.graph_norm:
            batch.x = self.norm_layer(batch.x, batch.batch)

        batch.x = self.norm_layer(batch.x)
        return batch
