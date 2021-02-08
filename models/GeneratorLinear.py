# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:29:57 2021

@author: othmane.mounjid
"""

import torch.nn as nn


### simple generator gan
class Generator_simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.in_numel = 1


    def forward(self, noise, labels = None):
        gen_input = noise 
        out = self.fc1(gen_input)
        return out
    