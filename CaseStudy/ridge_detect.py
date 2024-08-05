'''
input graph signals
1 --- 1 ---- 0 ---- 0
|     |      |      |
|     |      |      |
1 --- 1 ---- 0 ---- 0

target graph labels
0 --- 1 ---- 1 ---- 0
|     |      |      |
|     |      |      |
0 --- 1 ---- 1 ---- 0
|     |      |      |
|     |      |      |
0 --- 1 ---- 1 ---- 0
'''

import torch
from torch import nn


adj = [
    [0, 1, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1, 0]
]

adj = torch.Tensor(adj)

