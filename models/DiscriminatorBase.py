# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:31:04 2021

@author: othmane.mounjid
"""

import torch
import torch.nn as nn


class Discriminator_simple(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()        
        self.fc1 = nn.Linear(2, 1)
        self.sigm1 = nn.Linear(1, 2)


    def forward(self, x):
        y = x*x # y.shape
        z = torch.cat((x, y), 1) # z.shape
        validity = self.sigm1(self.fc1(z)) # validity.shape
        return validity