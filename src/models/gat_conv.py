import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing

from .mlp import MLP



class AGNNConv(MessagePassing):
    '''
    Additive form of GAT from DAGNN paper.

    In order to do the fair comparison with DeepSet. I add a FC-based layer before doing the attention.
    '''
    def __init__(self, in_channels, ouput_channels=None, wea=False, mlp=None, reverse=False):
        super(AGNNConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.wea = wea
        if self.wea:
            # fix the size of edge_attributes now
            self.edge_encoder = nn.Linear(16, ouput_channels)

        # linear transformation
        self.msg = MLP(in_channels, ouput_channels, ouput_channels, num_layer=3, p_drop=0.2) if mlp is None else mlp
        # self.msg = nn.Linear(in_channels, ouput_channels)

        # attention
        attn_dim = ouput_channels
        self.attn_lin = nn.Linear(ouput_channels + attn_dim, 1)


    # h_attn_q is needed; h_attn, edge_attr are optional (we just use kwargs to be able to switch node aggregator above)
    def forward(self, x, edge_index, edge_attr=None, **kwargs):

        # Step 2: Linearly transform node feature matrix.
        # h = self.msg(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        # h_i: query, h_j: key, value
        h_attn_q_i = self.msg(x_i)
        h_attn = self.msg(x_j)
        # see comment in above self attention why this is done here and not in forward
        if self.wea:
            edge_embedding = self.edge_encoder(edge_attr)
            h_attn = h_attn + edge_embedding    
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)
        t = h_attn * a_j
        return t

    def update(self, aggr_out):
        return aggr_out
