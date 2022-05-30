from typing import Optional, Callable, List
import os.path as osp
from sqlalchemy import false


import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils.data_utils import read_npz_file
from .load_data import circuit_parse_pyg


class CircuitDataset(InMemoryDataset):
    r"""
    A variety of circuit graph datasets, *e.g.*, open-sourced benchmarks,
    random circuits.
    Modified by Min.

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

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        name = "{}_{}_{}".format(self.args.num_gate_types, int(self.args.small_train), int(self.args.un_directed))
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.args.circuit_file, self.args.label_file]

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        circuits = read_npz_file(self.args.circuit_file, self.args.data_dir)['circuits'].item()
        labels = read_npz_file(self.args.label_file, self.args.data_dir)['labels'].item()
        
        if self.args.small_train:
            subset = self.args.small_train

        for cir_idx, cir_name in enumerate(circuits):
            print('Parse circuit: ', cir_name)
            x = circuits[cir_name]["x"]
            edge_index = circuits[cir_name]["edge_index"]
            y = labels[cir_name]["y"]
            graph = circuit_parse_pyg(x, edge_index, y, un_directed=self.args.un_directed, num_gate_types=self.args.num_gate_types, mask=self.args.mask)
            graph.name = cir_name
            data_list.append(graph)
            if self.args.small_train and cir_idx > subset:
                break


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'