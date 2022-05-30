'''
Implement two datasets for SAT problems. One for NeuronSAT and the other for CircuitSAT
'''

from typing import Optional, Callable, List
import os.path as osp
import random


import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils.sat_utils import gen_iclause_pair
from .load_data import aig_parse_pyg


class CircuitSATDataset(InMemoryDataset):
    r"""
    The PyG dataset for NeuroSATDataset.

    Args:
        root (string): Root directory where the dataset should be saved.
        args (object): The arguments specified by the main program.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = args.dataset
        self.args = args
        self.use_aig = args.use_aig
        self.min_n = args.min_n
        self.max_n = args.max_n
        self.exp_depth = args.exp_depth

        print('cnf format SR{} to SR{} problems.'.format(self.min_n, self.max_n))
        print('total # of problems: ', self.args.n_pairs)

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        name = "sr{}to{}_p{}_aig{}".format(self.args.min_n, self.args.max_n, self.args.n_pairs, int(self.use_aig))
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        # since the data is generated on the fly, we don't need the raw files here.
        return ['INFO']

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        '''
        The cnf dataset generation proecess followed by https://github.com/ryanzhangfan/NeuroSAT/blob/master/src/data_maker.py
        Here we ignore the constraint of `max_nodes_per_batch`.
        '''
        data_list = []

        n_cnt = self.args.max_n - self.args.min_n + 1
        problems_per_n = self.args.n_pairs * 1.0 / n_cnt

        for n_var in range(self.min_n, self.max_n+1):
            lower_bound = int((n_var - self.min_n) * problems_per_n)
            upper_bound = int((n_var - self.min_n + 1) * problems_per_n)
            for problems_idx in range(lower_bound, upper_bound):
                if (problems_idx % 1000) == 0:
                    print('generate {}/{} sat problems...'.format(problems_idx, self.args.n_pairs))
                n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(self.args, n_var)
                # use the sat problem directly here.
                # if random.random() < 0.5:
                #     iclauses.append(iclause_unsat)
                # else:
                iclauses.append(iclause_sat)
                n_clauses = len(iclauses)
                if self.use_aig:
                    graph = aig_parse_pyg(iclauses, n_vars, n_clauses, self.args.prob_loss)
                else:
                    raise("Unsupport")
                graph.problems_idx = problems_idx
                data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'