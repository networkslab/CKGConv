'''
        1 --- 0
       /       \
     /          \
    0            1
     \          /
      \        /
       1 --- 0
'''

import torch
from torch import nn
import torch_geometric as pyg

from torch_geometric.nn import Sequential
from torch_geometric.nn import GCNConv
from SimCKGConv import CKGConv
from tqdm import tqdm, trange

# Prepare Data

adj = [
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0]
]

x = torch.Tensor([1, 0, 0, 1, 1, 0]).view(-1, 1)
y = torch.LongTensor([1, 0, 0, 1, 1, 0]).view(-1, 1)

adj = torch.Tensor(adj)

edge_index, edge_attr = pyg.utils.dense_to_sparse(adj)


# data = pyg.Data(x=x, edge_index=edge_index, y=y)

deg = adj.sum(dim=1, keepdim=True)
rw = adj / deg.view(-1, 1)
out = rw
rrwp = [torch.eye(adj.size(0))]
for i in range(5):
    out = out @ rw
    rrwp.append(out)
rrwp = torch.stack(rrwp, dim=-1)
rrwp_index, rrwp_val = pyg.utils.dense_to_sparse(adj)

pe_dim = rrwp_val.size(1)



# -------- Config --------
num_layers = 4
in_dim = 1
out_dim = 1
hid_dim = 16
max_ep = 1000

device = 4

# ---------------- GCN -----------

GCN = [(nn.Linear(in_dim, hid_dim), 'x -> x')]
for i in range(num_layers):
    GCN += [(GCNConv(hid_dim, hid_dim, add_self_loops=True, normalize=True), 'x, edge_index -> x'),
            (nn.GELU(), 'x -> x')
            ]
GCN += [(nn.Linear(hid_dim, out_dim), 'x -> x'),
        (nn.Sigmoid, 'x -> x')]

GCN = Sequential('x, edge_index', GCN)

# # ------ Train GCN ----------
opt = torch.optim.Adam(GCN.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

print('--------- Train GCN -----------')
t_bar = trange(max_ep, desc='GCN', leave=True)

x, y = x.to(device), y.to(device)
edge_index = edge_index.to(device)
for ep in t_bar:
    opt.zero_grad()
    pred = GCN.forward(x, edge_index)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    pred_ = (pred >= 0.5).type(torch.long)
    acc = torch.mean((pred_ == y).type(torch.float))
    t_bar.set_description(f'GCN: bce_loss={loss.item()}  acc={acc.item()}')


# -------------- CKCCN ---------
CKGCN = [(nn.Linear(in_dim, hid_dim), 'x -> x')]
for i in range(num_layers):
    CKGCN += [(CKGConv(hid_dim, hid_dim, pe_dim, ffn_ratio=1., num_blocks=1), 'x, pe_index, pe_val -> x'),
            (nn.GELU(), 'x -> x')
            ]
CKGCN += [(nn.Linear(hid_dim, out_dim), 'x -> x'),
          (nn.Sigmoid, 'x -> x')]

CKGCN = Sequential('x, pe_index, pe_val -> x', CKGCN)









