#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvg_serial(w):
    w_avg = copy.deepcopy(w[-1])
    for k in w_avg.keys():
        # 减去最初的噪声再平均
        w_avg[k] = torch.div(w_avg[k]-w[0][k], len(w)-1)
    return w_avg
