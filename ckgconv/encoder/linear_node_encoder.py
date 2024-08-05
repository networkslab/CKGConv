import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder
from einops import rearrange, reduce, repeat


@register_node_encoder('LinearNode')
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(cfg.share.dim_in, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch



@register_node_encoder('LinearNodeV2')
class LinearNodeV2Encoder(torch.nn.Module):
    '''
        Not add to x; keep the original name
    '''
    def __init__(self, emb_dim, out_dim, attr_name='pe', **kwargs):
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
