import numpy as np
import random
import copy
import glob
import platform
import os
import argparse
import subprocess
import sys

import utils.sat_utils as sat_utils
import utils.circuit_utils as circuit_utils

def get_parse_args():
    parser = argparse.ArgumentParser(description='SAT script')

    # SAT
    parser.add_argument('--min_n', type=int, default=10, help='SR Min')
    parser.add_argument('--max_n', type=int, default=10, help='SR Max')
    parser.add_argument('--pc', type=int, default=10000, help='Simulation Pattern Counts')
    parser.add_argument('--n_pairs', type=int, default=10, help='No of Pairs/Circuits')
    parser.add_argument('--start_idx', type=int, default=0, help='Circuit Start Index')

    parser.add_argument('--cv_ratio', type=int, default=-1, help='Clause to Varables')
    parser.add_argument('--p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', type=float, default=0.4)

    parser.add_argument('--sat_folder', type=str, default='../dataset/sat_default/')

    parser.add_argument('--testset', action='store_true', default=False)

    sat_args = parser.parse_args()
    sat_args.gate_to_index = {"INPUT": 0, "AND": 1, "NAND": 2, "OR": 3, "NOR": 4, "NOT": 5, "XOR": 6}

    return sat_args

if __name__ == '__main__':
    sat_args = get_parse_args()
    n_cnt = sat_args.max_n - sat_args.min_n + 1
    problems_per_n = sat_args.n_pairs * 1.0 / n_cnt
    graphs = {}
    labels = {}
    tot_random_succ_r = []
    all_cnt = 0
    solved_cnt = 0

    for n_var in range(sat_args.min_n, sat_args.max_n+1):
        lower_bound = int((n_var - sat_args.min_n) * problems_per_n)
        upper_bound = int((n_var - sat_args.min_n + 1) * problems_per_n)
        problems_idx = lower_bound + sat_args.start_idx
        while problems_idx < upper_bound + sat_args.start_idx:
            if (problems_idx % 1000) == 0:
                print('generate {}/{} sat problems...'.format(problems_idx, sat_args.n_pairs))
            n, iclauses, iclause_unsat, iclause_sat, assignment = sat_utils.gen_iclause_pair(sat_args, n_var)
            x_data, edge_index = sat_utils.cnf_to_netlist(iclauses, n_var)
            if len(x_data) == 0:
                continue

            # Generate feature
            x_data, edge_data, level_list, fanin_list, fanout_list = circuit_utils.parse_x_data(x_data, edge_index)
            PI_list = level_list[0]
            # assert len(level_list[-1])==1, 'Error in PO'
            # assert len(PI_list) == n_var, 'Error in PI'
            if len(level_list[-1])!= 1:
                continue
            if len(PI_list) != n_var:
                continue

            # Generate x_data
            PO_idx = level_list[-1][0]
            x_data = circuit_utils.generate_prob_cont(x_data, PI_list, level_list, fanin_list)
            x_data = circuit_utils.generate_prob_obs(x_data, level_list, fanin_list, fanout_list)
            x_data, _ = circuit_utils.identify_reconvergence(x_data, level_list, fanin_list, fanout_list)
            
            # Generate Mask 
            print('='*20)
            y = [0.5] * len(x_data)
            mask_list = [PO_idx]
            mask_x_data = copy.deepcopy(x_data)
            for idx, x_data_info in enumerate(mask_x_data):
                mask_x_data[idx].append(-1)

            for mask_cnt in range(-1, len(PI_list)+1):
                # 1. Mask Generation 
                if mask_cnt == -1:
                    print('[INFO] No mask')
                elif mask_cnt == 0:
                    mask_x_data[PO_idx][-1] = 1
                    print('[INFO] Mask PO')
                else:
                    PI_distance = []
                    for PI_idx in PI_list:
                        if PI_idx in mask_list:
                            PI_distance.append(0.5)
                        elif y[PI_idx] < 0.5:
                            PI_distance.append(y[PI_idx])
                        else:
                            PI_distance.append(1-y[PI_idx])
                    sorted_idx = sorted(range(len(PI_distance)), key=lambda k: PI_distance[k], reverse=True)[-1]
                    PI_idx = PI_list[sorted_idx]
                    if PI_idx in mask_list:
                        break
                    print('Mask PI {:}, Uncertainty Score: {:}'.format(PI_idx, PI_distance[sorted_idx]))
                    mask_list.append(PI_idx)
                    mask_x_data[PI_idx][-1] = int(y[PI_idx] > 0.5)
                
                # 2. Simulation with mask
                if sat_args.testset:
                    success, y = circuit_utils.search_solution(mask_x_data, PI_list, level_list, fanin_list, list(assignment))
                else:
                    success, y, succ_r = circuit_utils.mask_simulator(mask_x_data, PI_list, level_list, fanin_list, sat_args.pc)

                # 3. Save
                if success > 0:
                    if mask_cnt == -1:
                        circuit_name = 'SR{:}_CV{:}_{:}'.format(n_var, sat_args.cv_ratio, problems_idx)
                        problems_idx += 1
                    else:
                        circuit_name = 'SR{:}_CV{:}_{:}_Mask{:}'.format(n_var, sat_args.cv_ratio, problems_idx, mask_cnt)
                    new_y_data = []
                    for idx in range(len(y)):
                        new_y_data.append([y[idx]])
                    graphs[circuit_name] = {'x': np.array(mask_x_data).astype('float32'), 'edge_index': np.array(edge_data)}
                    labels[circuit_name] = {'y': np.array(new_y_data).astype('float32')}
                    # when generate test dataset, no iterative mask 
                    if sat_args.testset:
                        print('[INFO] Only generate test dataset')
                        break
                    
                # 4. Print
                if not success:
                    print('[INFO] Solving failed')
                    break
                else:
                    if mask_cnt != -1:
                        print('[INFO] Save ', circuit_name, ': ', mask_list[:mask_cnt+1])
                        print('Random Pattern Successful Ratio: {:}%'.format(succ_r*100))
                        if len(mask_list) == n_var + 1:
                            print('Reach maximum mask count')
                            print('[SUCCESS] Solved')
                            solved_cnt += 1
                            break
                    else:
                        print('[INFO] Save No Mask ')

                print('-'*10)
            print()

            all_cnt += 1

    if not os.path.exists(sat_args.sat_folder):
        os.makedirs(sat_args.sat_folder)

    if sat_args.cv_ratio > 0:
        circuits_file = sat_args.sat_folder + '/sr{:}_{:}_CV{:}_{:}_graphs.npz'.format(sat_args.min_n, sat_args.max_n, sat_args.cv_ratio, sat_args.n_pairs)
        labels_file = sat_args.sat_folder + '/sr{:}_{:}_CV{:}_{:}_labels.npz'.format(sat_args.min_n, sat_args.max_n, sat_args.cv_ratio, sat_args.n_pairs)
    else:
        circuits_file = sat_args.sat_folder + './npz/sr{:}_{:}_full_{:}_graphs.npz'.format(sat_args.min_n, sat_args.max_n, sat_args.n_pairs)
        labels_file = sat_args.sat_folder + './npz/sr{:}_{:}_full_{:}_labels.npz'.format(sat_args.min_n, sat_args.max_n, sat_args.n_pairs)
    
    np.savez_compressed(circuits_file, circuits=graphs)
    np.savez_compressed(labels_file, labels=labels)
    
    print('[INFO] # Graphs: ', len(graphs))
    if not sat_args.testset:
        print('Solved {:}/{:}={:}%'.format(solved_cnt, all_cnt, solved_cnt/all_cnt*100))

