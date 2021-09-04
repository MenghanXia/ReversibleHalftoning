#!/usr/bin/env python
"""
-------------------------------------------------
   File Name：   dct
   Author :      wenbo
   date：         12/4/2019
   Description :
-------------------------------------------------
   Change Activity:
                   12/4/2019:
-------------------------------------------------
"""
__author__ = 'wenbo'

from torch import nn
from ._dct import LinearDCT, apply_linear_2d


class DCT_Lowfrequency(nn.Module):
    def __init__(self, size=256, fLimit=50):
        super(DCT_Lowfrequency, self).__init__()
        self.fLimit = fLimit
        self.dct = LinearDCT(size, type='dct', norm='ortho')
        self.dctTransformer = lambda x: apply_linear_2d(x, self.dct)

    def forward(self, x):
        x = self.dctTransformer(x)
        x = x[:, :, :self.fLimit, :self.fLimit]
        return x
