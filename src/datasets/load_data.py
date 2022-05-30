import subprocess
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.utils.convert import to_networkx

from .ordered_data import OrderedData
from utils.dag_utils import return_order_info
from utils.data_utils import construct_node_feature, add_skip_connection, add_edge_attr, one_hot

from utils.sat_utils import solve_sat, recursion_generation, one_hot_gate_type, write_dimacs_to
from utils.circuit_utils import aig_simulation


def circuit_parse_pyg(x, edge_index, y, no_node_cop=True, node_reconv=False, un_directed=False, num_gate_types=9, mask=False):
    '''
    A function to parse the circuits and labels stored in `.npz` format to `Pytorch Geometric` Data.
    Optional, will add the skip connection, and the edge attributes into the graphs if specified.
    Modified by Min.
    ...
    Parameters:
        x : numpy array with shape of [num_nodes, 9], the node feature matrix with shape [num_nodes, num_node_features], the current dimension of num_node_features is 9, wherein 0th - node_name defined in bench (str); 1st - integer index for the gate type; 2nd - logic level; 3rd - C1; 4th - C0; 5th - Obs; 6th - fan-out, 7th - boolean recovengence, 8th - index of the source node (-1 for non recovengence), 9th - masked or not and the masked value (-1, no masked). 
        edge_index : numpy array with shape of [num_edges, 2], thhe connectivity matrix.
        use_edge_attr : bool, whether to use the edge attributes.
        reconv_skip_connection: bool, whether to add the skip connection between source nodes and reconvergence nodes.
        logic_diff_embedding: str, the way to encode the discrete logic level.
        predict_diff : bool, whether to predict the difference between the simulated ground-truth probability and C1.
        diff_multiplier : int, the multiplier for the difference between the simulated ground-truth probability and C1.
        node_cop : bool, whether to use the C1 values as the node features.
        node_reconv : bool, whether to use the reconvergence info as the node features.
        dim_edge_feature : int, the dimension of node features.
        logic_implication: bool, whether to use the logic implication as the node feature or not.
        mask: bool, whether to use the masking of node embedding or not.
    Return:
        graph : torch_geometric.data.Data, the constructed pyG data.
    '''
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)
    y_torch = torch.tensor(y, dtype=torch.float)


    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_index = edge_index.t().contiguous()
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))


    graph = OrderedData(x=x_torch, edge_index=edge_index, forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index)
    graph.y = y_torch
    graph.use_edge_attr = False

    # add C1 info
    graph.c1 = torch.tensor(x[:, 3:4], dtype=torch.float)
    # add gt info
    graph.gt = torch.tensor(y, dtype=torch.float)
    # add indices for gate types
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)

    if mask:
        graph.mask = torch.tensor(x[:, 9:], dtype=torch.float)

    if un_directed:
        graph = ToUndirected()(graph)
    return graph


def cnf_parse_pyg(iclauses, n_vars, n_clauses, n_nodes):
    '''
    A function to parse the cnf `Pytorch Geometric` Data.
        edge_index should be directed edges from literals to clauses.
        x: 0 - literals and their negations; 1 - clauses
        n_vars: the numbers of literals in the problem
        n_nodes: the total numbers of nodes in the problem (2*n_vars + n_clauses)
    '''
    assert n_nodes == (2 * n_vars + n_clauses), 'check the size of literals/clauses in the problem.'
    # construct x
    x = torch.zeros(size=(n_vars*2, 1))
    x = torch.cat([x, torch.ones(size=(n_clauses, 1))], dim=0)
    # construct edge_index
    edge_index = []
    for idx, iclause in enumerate(iclauses):
        c_idx = n_vars * 2 + idx
        for i in iclause:
            l_idx = (i - 1) if i > 0 else (abs(i) - 1 + n_vars)
            edge_index.append([l_idx, c_idx])
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    is_sat, sol = solve_sat(n_vars, iclauses)

    if is_sat:
        y = torch.tensor([1], dtype=torch.float)
        sol = torch.tensor(sol, dtype=torch.float)
        sol = torch.cat([sol, torch.zeros(n_nodes - n_vars)], dim=0)
    else:
        y = torch.tensor([0])
        sol = torch.zeros(n_nodes)
    graph = Data(x=x, edge_index=edge_index.t().contiguous(), y=y, sol=sol, n_vars=n_vars, n_clauses=n_clauses, n_nodes=n_nodes)

    return graph

def aig_parse_pyg(iclauses, n_vars, n_clauses, prob_loss):
    '''
    A function to parse the cnf to aig, then to circuit to `Pytorch Geometric` Data.
    Input:
        iclause: clauses list
        n_vars: number of variables
        n_clauses: number of clauses
    Return:
        x: one_hot encoding of [PI, AND, NOT]
        edge_index: edge connection pairs: each pair [x, y] from x to y
    For AIG, the nodes can be categorized as the Literal node, internal AND nodes, internal NOT node. The type values for each kind of nodes are as follows:
        * Literal input node: 0;
        * Internal AND nodes: 1;
        * Internal NOT nodes: 2;
    '''
    # step 1: store dimacs format
    dimacs_tmp = '/tmp/sat.dimacs'
    write_dimacs_to(n_vars, iclauses, dimacs_tmp)
    # step 2: dimacs to aig
    aig_tmp = '/tmp/sat.aig'
    subprocess.call(["./external/aiger/cnf2aig/cnf2aig", dimacs_tmp, aig_tmp])
    # step 3: aig to abc opimized aig
    aig_abc_tmp = '/tmp/aig_abc.aig'
    subprocess.call(["./external/abc/abc", "-c", "r %s; b; ps; b; rw -l; rw -lz; b; rw -lz; b; ps; cec; w %s" % (aig_tmp, aig_abc_tmp)])
    # step 4: aig to aag
    aag_abc_tmp = '/tmp/aig_abc.aag'
    subprocess.call(["./external/aiger/aiger/aigtoaig", aig_abc_tmp, aag_abc_tmp])
    # step 4: read aag
    with open(aag_abc_tmp, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(" ")
    assert header[0] == 'aag', 'The header of AIG file is wrong.'
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    assert n_outputs == 1, 'The AIG has multiple outputs.'
    assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x = []
    edge_index = []
    # node_labels = []
    not_dict = {}
    
    # Add Literal node
    for i in range(n_inputs):
        x += [one_hot(0, 3)]
        # node_labels += [0]

    # Add AND node
    for i in range(n_inputs+1, n_inputs+1+n_and):
        x += [one_hot(1, 3)]
        # node_labels += [1]


    # sanity-check
    for (i, line) in enumerate(lines[1:1+n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1+n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2 - 1

    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2+n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'
    # finish sanity-check

    # Add edge
    for (i, line) in enumerate(lines[2+n_inputs: 2+n_inputs+n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2 - 1
        # assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'

        # 1. First edge
        input1_idx = int(line[1]) // 2 - 1
        sign1_idx = int(line[1]) % 2
        # If there's a NOT node
        if sign1_idx == 1:
            if input1_idx in not_dict.keys():
                not_idx = not_dict[input1_idx]
            else:
                x += [one_hot(2, 3)]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input1_idx] = not_idx
                edge_index += [[input1_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input1_idx, output_idx]]


        # 2. Second edge
        input2_idx = int(line[2]) // 2 - 1
        sign2_idx = int(line[2]) % 2
        # If there's a NOT node
        if sign2_idx == 1:
            if input2_idx in not_dict.keys():
                not_idx = not_dict[input2_idx]
            else:
                x += [one_hot(2, 3)]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input2_idx] = not_idx
                edge_index += [[input2_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input2_idx, output_idx]]
    
    
    if sign_final == 1:
        x += [one_hot(2, 3)]
        # node_labels += [2]
        not_idx = len(x) - 1
        edge_index += [[index_final_and, not_idx]]
    
    # simulation
    if prob_loss:
        y_prob = aig_simulation(x, edge_index)
        y_prob = torch.tensor(y_prob, dtype=torch.float)

    x = torch.cat(x, dim=0).float()
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    y = torch.tensor([1])

    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x.size(0))
    
    graph = OrderedData(x=x, edge_index=edge_index, forward_level=forward_level, forward_index=forward_index, 
    backward_level=backward_level, backward_index=backward_index)
    
    # Draw graph
    # nx_graph = to_networkx(graph)

    # import matplotlib.pyplot as plt
    # plt.figure(1,figsize=(14,12)) 
    # nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_color = np.array(node_labels), node_size=75,linewidths=6)
    # plt.show()

    graph.y, graph.n_vars, graph.n_clauses = y, n_vars, n_clauses
    graph.y_prob = y_prob

    return graph