import math
from functools import partial

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from einops import rearrange

from src.utils import instantiate
import src.utils.registry as registry

from src.models.sequence.fno.tfno import FNO1d

class FNOBase(nn.Module):
    def __init__(
        self,
        n_modes_height,
        hidden_channels,
        in_channels=3, 
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        incremental_n_modes=None,
        n_layers=4,
        non_linearity=F.gelu,
        use_mlp=False, mlp=None,
        norm=None,
        skip='soft-gating',
        separable=False,
        preactivation=False,
        factorization=None, 
        rank=1.0,
        joint_factorization=False, 
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode='one-sided',
        fft_norm='forward',
        **kwargs):
        super().__init__()
        self.model = FNO1d(
            n_modes_height=n_modes_height,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels) # takes in (batch x 1 x seqlen)

    def forward(self, input_ids, position_ids=None, state=None): # state for the repo interface
        hidden_states = self.model(input_ids.transpose(1,2)).transpose(1,2)
        return hidden_states, None
