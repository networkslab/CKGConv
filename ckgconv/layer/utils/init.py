import torch
from torch import nn
import math
from timm.models.layers import trunc_normal_, variance_scaling_, trunc_normal_tf_
from torch.nn.init import _calculate_fan_in_and_fan_out


def trunc_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        trunc_normal_(m.weight, std=0.02)
    elif isinstance(m, nn.Parameter):
        trunc_normal_(m, std=.02)



def trunc_xavier_normal_init_(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        variance_scaling_(m.weight, scale=1.0,
                          mode='fan_avg', distribution='truncated_normal')
    elif isinstance(m, nn.Parameter):
        variance_scaling_(m, scale=1.0,
                          mode='fan_avg', distribution='truncated_normal')


def trunc_kaiming_normal_init_(m, scale=1.0):
    constant = .87962566103423978
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
        denom = fan_in / 2
        variance = scale / denom
        trunc_normal_tf_(m.weight, std=math.sqrt(variance)/constant)
    elif isinstance(m, nn.Parameter):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(m)
        denom = fan_in / 2
        variance = scale / denom
        trunc_normal_tf_(m, std=math.sqrt(variance)/constant)
