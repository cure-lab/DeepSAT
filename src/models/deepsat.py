from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from utils.dag_utils import subgraph, custom_backward_subgraph

from .gat_conv import AGNNConv
from .gcn_conv import AggConv
from .deepset_conv import DeepSetConv
from .gated_sum_conv import GatedSumConv
from .mlp import MLP
from .layernorm_gru import LayerNormGRU
from .layernorm_lstm import LayerNormLSTM

from torch.nn import LSTM, GRU


_aggr_function_factory = {
    'agnnconv': AGNNConv,
    'deepset': DeepSetConv,
    'gated_sum': GatedSumConv,
    'conv_sum': AggConv,
}

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
    'layernorm_lstm': LayerNormLSTM,
    'layernorm_gru': LayerNormGRU,
}


class DeepSAT(nn.Module):
    '''
    DeepSAT Graph Neural Networks for Satifiability problems.
    '''
    def __init__(self, args):
        super(DeepSAT, self).__init__()
        
        self.args = args

        # configuration
        self.num_rounds = args.num_rounds
        self.device = args.device
        self.reverse = args.reverse
        self.mask = args.mask

        # dimensions
        self.num_aggr = args.num_aggr
        self.dim_node_feature = args.dim_node_feature
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.dim_pred = args.dim_pred
        self.num_fc = args.num_fc
        self.wx_update = args.wx_update
        self.wx_mlp = args.wx_mlp
        self.dim_edge_feature = args.dim_edge_feature

        # 1. message/aggr-related
        dim_aggr = self.dim_hidden
        if self.args.aggr_function in _aggr_function_factory.keys():
            self.aggr_forward = _aggr_function_factory[self.args.aggr_function](dim_aggr, self.dim_hidden)
            if self.reverse:
                self.aggr_backward = _aggr_function_factory[self.args.aggr_function](dim_aggr, self.dim_hidden, reverse=True)
        else:
            raise KeyError('no support {} aggr function.'.format(self.args.aggr_function))


        # 2. update-related
        if self.args.update_function in _update_function_factory.keys():
            # Here only consider the inputs as the concatenated vector from embedding and feature vector.
            if self.wx_update:
                self.update_forward = _update_function_factory[self.args.update_function](self.dim_node_feature+self.dim_hidden, self.dim_hidden)
                if self.reverse:
                    self.update_backward = _update_function_factory[self.args.update_function](self.dim_node_feature+self.dim_hidden, self.dim_hidden)
            else:
                self.update_forward = _update_function_factory[self.args.update_function](self.dim_hidden, self.dim_hidden)
                if self.reverse:
                    self.update_backward = _update_function_factory[self.args.update_function](self.dim_hidden, self.dim_hidden)
        else:
            raise KeyError('no support {} update function.'.format(self.args.update_function))
        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        self.emd_int = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False


        # 3. predictor-related
        # TODO: support multiple predictors. Use a nn.ModuleList to handle it.
        self.norm_layer = args.norm_layer
        self.activation_layer = args.activation_layer
        if self.wx_mlp:
            self.predictor = MLP(self.dim_hidden+self.dim_node_feature, self.dim_mlp, self.dim_pred, 
            num_layer=self.num_fc, norm_layer=self.norm_layer, act_layer=self.activation_layer, sigmoid=False, tanh=False)
        else:
            self.predictor = MLP(self.dim_hidden, self.dim_mlp, self.dim_pred, 
            num_layer=self.num_fc, norm_layer=self.norm_layer, act_layer=self.activation_layer, sigmoid=False, tanh=False)

        

    def forward(self, G):
        num_nodes = G.num_nodes
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        one = self.one
        h_init = self.emd_int(one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        h_init = h_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)
        # h_init = torch.empty(1, num_nodes, self.dim_hidden).to(self.device)
        # nn.init.normal_(h_init)

        if self.mask:
            h_true = torch.ones_like(h_init).to(self.device)
            h_false = -torch.ones_like(h_init).to(self.device)
            h_true.requires_grad = False
            h_false.requires_grad = False

            h_init = self.imply_mask(G, h_init, h_true, h_false)
        else:
            h_true = None
            h_false = None

        if 'lstm' in self.args.update_function:
            preds = self._lstm_forward(G, h_init, num_layers_f, num_layers_b, num_nodes, h_true, h_false)
        elif 'gru' in self.args.update_function:
            preds = self._gru_forward(G, h_init, num_layers_f, num_layers_b, h_true, h_false)
        else:
            raise NotImplementedError('The update function should be specified as one of lstm and gru.')
        
        return preds
            
    
    def _lstm_forward(self, G, h_init, num_layers_f, num_layers_b, num_nodes, h_true=None, h_false=None):
        x, edge_index = G.x, G.edge_index
        edge_attr = None
        
        node_state = (h_init, torch.zeros(1, num_nodes, self.dim_hidden).to(self.device)) # (h_0, c_0). here we only initialize h_0. TODO: option of not initializing the hidden state of LSTM.
        

        for _ in range(self.num_rounds):
            for l_idx in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == l_idx
                l_node = G.forward_index[layer_mask]

                l_state = (torch.index_select(node_state[0], dim=1, index=l_node), 
                            torch.index_select(node_state[1], dim=1, index=l_node))

                l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=1)
                msg = self.aggr_forward(node_state[0].squeeze(0), l_edge_index, l_edge_attr)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                l_x = torch.index_select(x, dim=0, index=l_node)
                
                if self.args.wx_update:
                    _, l_state = self.update_forward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                else:
                    _, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)

                node_state[0][:, l_node, :] = l_state[0]
                node_state[1][:, l_node, :] = l_state[1]

                if self.mask:
                    node_state[0][:] = self.imply_mask(G, node_state[0], h_true, h_false)

            if self.reverse:
                for l_idx in range(1, num_layers_b):
                    # backward layer
                    layer_mask = G.backward_level == l_idx
                    l_node = G.backward_index[layer_mask]
                    
                    l_state = (torch.index_select(node_state[0], dim=1, index=l_node), 
                                torch.index_select(node_state[1], dim=1, index=l_node))
                    
                    l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=0)
                    msg = self.aggr_backward(node_state[0].squeeze(0), l_edge_index, l_edge_attr)
                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    l_x = torch.index_select(x, dim=0, index=l_node)
                    
                    if self.args.wx_update:
                        _, l_state = self.update_backward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                    else:
                        _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)
                    
                    node_state[0][:, l_node, :] = l_state[0]
                    node_state[1][:, l_node, :] = l_state[1]

                    if self.mask:
                        node_state[0][:] = self.imply_mask(G, node_state[0], h_true, h_false)
               

        node_embedding = node_state[0].squeeze(0)
        if self.wx_mlp:
            pred = self.predictor(torch.cat([node_embedding, x], dim=1))
        else:
            pred = self.predictor(node_embedding)

        return pred
    
    def _gru_forward(self, G, h_init, num_layers_f, num_layers_b, h_true=None, h_false=None):
        x, edge_index = G.x, G.edge_index
        edge_attr = None

        node_state = h_init # (h_0). here we initialize h_0. TODO: option of not initializing the hidden state of GRU.


        for _ in range(self.num_rounds):
            for l_idx in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == l_idx
                l_node = G.forward_index[layer_mask]
                
                l_state = torch.index_select(node_state, dim=1, index=l_node)

                l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=1)
                msg = self.aggr_forward(node_state.squeeze(0), l_edge_index, l_edge_attr)
                l_msg = torch.index_select(msg, dim=0, index=l_node)
                l_x = torch.index_select(x, dim=0, index=l_node)
                
                if self.args.wx_update:
                    _, l_state = self.update_forward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                else:
                    _, l_state = self.update_forward(l_msg.unsqueeze(0), l_state)
                node_state[:, l_node, :] = l_state
                
                if self.mask:
                    node_state = self.imply_mask(G, node_state, h_true, h_false)
            
            if self.reverse:
                for l_idx in range(1, num_layers_b):
                    # backward layer
                    layer_mask = G.backward_level == l_idx
                    l_node = G.backward_index[layer_mask]
                    
                    l_state = torch.index_select(node_state, dim=1, index=l_node)

                    l_edge_index, l_edge_attr = subgraph(l_node, edge_index, edge_attr, dim=0)
                    msg = self.aggr_backward(node_state.squeeze(0), l_edge_index, l_edge_attr)
                    l_msg = torch.index_select(msg, dim=0, index=l_node)
                    l_x = torch.index_select(x, dim=0, index=l_node)
                    
                    if self.args.wx_update:
                        _, l_state = self.update_backward(torch.cat([l_msg, l_x], dim=1).unsqueeze(0), l_state)
                    else:
                        _, l_state = self.update_backward(l_msg.unsqueeze(0), l_state)                
                    
                    node_state[:, l_node, :] = l_state

                    if self.mask:
                        node_state = self.imply_mask(G, node_state, h_true, h_false)


        node_embedding = node_state.squeeze(0)

        if self.wx_mlp:
            pred = self.predictor(torch.cat([node_embedding, x], dim=1))
        else:
            pred = self.predictor(node_embedding)        
        return pred 
    
    def imply_mask(self, G, h, h_true, h_false):
        # logic implication using masking
        true_mask = (G.mask == 1.0).unsqueeze(0)
        false_mask = (G.mask == 0.0).unsqueeze(0)
        normal_mask = (G.mask == -1.0).unsqueeze(0)
        h_mask = h * normal_mask + h_true * true_mask + h_false * false_mask
        return h_mask



def get_deepsat_gnn(args):
    return DeepSAT(args)
