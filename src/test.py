from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import torch

from config import get_parse_args
from utils.logger import Logger
from utils.random_seed import set_seed
from utils.sat_utils import solve_sat_iteratively
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory


def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str

    print(args)
    args.num_rounds = args.test_num_rounds
    Logger(args)

    dataset = dataset_factory[args.dataset](args.data_dir, args)
    # Do the shuffle
    # perm = torch.randperm(len(dataset))
    # dataset = dataset[perm]
    # split = args.test_split
    # dataset = dataset[:100]
    data_len = len(dataset)
    print('Total # Test SAT problems: ', data_len)


    detector = detector_factory['base'](args)

    print('Start Solving the SAT problem using DeepGate with Logic Implication...')
    
    correct = 0
    total = 0
    for ind, g in enumerate(dataset):
        if 'Mask' in g.name:
            continue
        total += 1
        sol, sat = solve_sat_iteratively(g, detector)

        print('# {} SAT: '.format(ind),sat)

        if sat:
            correct +=1

    print('ACC: {:.2f}% ({}/{})'.format(100*correct/total, correct, total))


if __name__ == '__main__':
    args = get_parse_args()
    set_seed(args)

    test(args)
