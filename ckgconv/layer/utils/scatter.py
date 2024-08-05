from functools import partial


from torch_geometric.utils import scatter



scatter_add = partial(scatter, reduce='add')
scatter_max = partial(scatter, reduce='max')
scatter_mul = partial(scatter, reduce='mul')