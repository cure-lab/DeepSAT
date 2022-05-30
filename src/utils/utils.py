from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import math
import copy

from .circuit_utils import random_pattern_generator, logic

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count


class custom_DataParallel(nn.parallel.DataParallel):
# define a custom DataParallel class to accomodate igraph inputs
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(custom_DataParallel, self).__init__(module, device_ids, output_device, dim)

    def scatter(self, inputs, kwargs, device_ids):
        # to overwride nn.parallel.scatter() to adapt igraph batch inputs
        G = inputs[0]
        scattered_G = []
        n = math.ceil(len(G) / len(device_ids))
        mini_batch = []
        for i, g in enumerate(G):
            mini_batch.append(g)
            if len(mini_batch) == n or i == len(G)-1:
                scattered_G.append((mini_batch, ))
                mini_batch = []
        return tuple(scattered_G), tuple([{}]*len(scattered_G))


def collate_fn(G):
    return [copy.deepcopy(g) for g in G]

def pyg_simulation(g, pattern=[]):
    # PI, Level list
    max_level = 0
    PI_indexes = []
    fanin_list = []
    for idx, ele in enumerate(g.forward_level):
        level = int(ele)
        fanin_list.append([])
        if level > max_level:
            max_level = level
        if level == 0:
            PI_indexes.append(idx)
    level_list = []
    for level in range(max_level + 1):
        level_list.append([])
    for idx, ele in enumerate(g.forward_level):
        level_list[int(ele)].append(idx)
    # Fanin list 
    for k in range(len(g.edge_index[0])):
        src = g.edge_index[0][k]
        dst = g.edge_index[1][k]
        fanin_list[dst].append(src)
    
    ######################
    # Simulation
    ######################
    y = [0] * len(g.x)
    if len(pattern) == 0:
        pattern = random_pattern_generator(len(PI_indexes))
    j = 0
    for i in PI_indexes:
        y[i] = pattern[j]
        j = j + 1
    for level in range(1, len(level_list), 1):
        for node_idx in level_list[level]:
            source_signals = []
            for pre_idx in fanin_list[node_idx]:
                source_signals.append(y[pre_idx])
            if len(source_signals) > 0:
                if int(g.x[node_idx][1]) == 1:
                    gate_type = 1
                elif int(g.x[node_idx][2]) == 1:
                    gate_type = 5
                else:
                    raise("This is PI")
                y[node_idx] = logic(gate_type, source_signals)

    # Output
    if len(level_list[-1]) > 1:
        raise('Too many POs')
    return y[level_list[-1][0]], pattern
