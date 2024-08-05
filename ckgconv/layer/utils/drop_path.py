import torch
from torch import nn
from timm.models.layers import DropPath




class GraphDropPath(nn.Module):
    def __init__(self, drop_prob: float=0., scale_by_keep: bool=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x, index=None, num_graphs=None):
        if self.drop_prob == 0. or not self.training:
            return x

        if index is None:
            num_graphs = 1
            index = x.new_zeros(1, dtype=torch.long)
        else:
            num_graphs = num_graphs if num_graphs is not None else torch.max(index) + 1

        keep_prob = 1 - self.drop_prob
        shape = (num_graphs, ) + (1, ) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        random_tensor = random_tensor[index]

        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


