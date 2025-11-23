#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import os
from config import Config


def Dwt(f, depth=None):
    x0 = (f[:, ::2] + f[:, 1::2])/2
    x1 = (f[:, 1::2] - f[:, ::2])/2
    return x0, x1


def Iwt(x0, x1, depth=None):
    rows, cols_x0 = x0.shape
    _, cols_x1 = x1.shape
    g = torch.zeros((rows, cols_x0 + cols_x1), dtype=x0.dtype, device=x0.device)

    g[:, 1::2] = x0 + x1
    g[:, ::2] = x0 - x1
    return g


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)

    return data
