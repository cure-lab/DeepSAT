'''
source: https://github.com/emreyolcu/sat/blob/master/code/kcolor.py
Author: emreyolcu

Modified by lee-man

`pip install cnfgen` before using this code 
'''
from __future__ import print_function

import argparse
import os
import random
import subprocess
import copy
import numpy as np

import utils.sat_utils
import utils.circuit_utils
from utils.graph_coloring import *

import sys
import external.PyMiniSolvers.minisolvers as minisolvers

def create_sat_problem(filename, n, p, k):
    if os.getcwd()[-3:] == 'src':
        external_folder = './external/'
    elif os.getcwd()[-7:] == 'deepsat':
        external_folder = './src/external/'
    else:
        raise('Wrong file path')
    
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp/')

    while True:
        subprocess.call(['cnfgen', '-q', '-o', './tmp/tmp.cnf', 'kcolor', str(k), 'gnp', str(n), str(p)])
        
        solver = minisolvers.MinisatSolver()
        # Read CNF
        f = open('./tmp/tmp.cnf')
        lines = f.readlines()
        iclauses = []
        for idx, line in enumerate(lines):
            if idx == 0:
                arr = line.replace('\n', '').split(' ')
                n_var = int(arr[2])
                n_clause = int(arr[3])
            else:
                arr = line.replace('\n', '').split(' ')
                iclause = []
                for ele in arr[:-1]:
                    iclause.append(int(ele))
                iclauses.append(iclause)

        # Check SAT
        for i in range(n_var):
            solver.new_var(dvar=True)
        for c in iclauses:
            solver.add_clause(c)

        if solver.solve(): 
            assignment = solver.get_model()
            x_data, edge_index = sat_utils.cnf_to_netlist(iclauses, n_var)

            # Generate feature
            x_data, edge_data, level_list, fanin_list, fanout_list = circuit_utils.parse_x_data(x_data, edge_index)
            PI_list = level_list[0]
            if len(level_list[-1])!= 1:
                continue
            if len(PI_list) != n_var:
                continue
            
            # Generate x_data
            PO_idx = level_list[-1][0]
            x_data = circuit_utils.generate_prob_cont(x_data, PI_list, level_list, fanin_list)
            x_data = circuit_utils.generate_prob_obs(x_data, level_list, fanin_list, fanout_list)
            x_data, _ = circuit_utils.identify_reconvergence(x_data, level_list, fanin_list, fanout_list)
            mask_x_data = copy.deepcopy(x_data)
            for idx, x_data_info in enumerate(mask_x_data):
                mask_x_data[idx].append(-1)
            
            success, y = circuit_utils.search_solution(mask_x_data, PI_list, level_list, fanin_list, list(assignment))

            if success: 
                new_y_data = []
                for idx in range(len(y)):
                    new_y_data.append([y[idx]])
                return mask_x_data, edge_data, new_y_data


        os.remove('./tmp/tmp.cnf')

def create_kcolor_problem(no_var, edge_ratio):
    while (True):
        # Problem generation
        adj_vec_list = []
        for src in range(no_var):
            for dst in range(src+1, no_var):
                adj_vec_list.append(0)
        for i in range(int(len(adj_vec_list) * edge_ratio)):
            adj_vec_list[i] = 1
        random.shuffle(adj_vec_list)

        edges = []
        tp = 0
        for src in range(no_var):
            for dst in range(src+1, no_var):
                if adj_vec_list[tp] == 1:
                    edges.append([src, dst])
                tp += 1
        matrix = np.zeros((no_var, no_var))
        for edge in edges:
            matrix[edge[0]-1][edge[1]-1] = 1
            matrix[edge[1]-1][edge[0]-1] = 1
        
        # Problem to CNF
        objGrafiNeSat = GrafiNeSat(matrix, 'test.cnf')
        lines = objGrafiNeSat.cnf_lines.split('\n')

        clauses = []
        for line in lines[:-1]:
            ele_list = line.split(' ')
            clause = []
            for ele in ele_list:
                if int(ele) == 0:
                    break
                clause.append(int(ele))
            if len(clause) > 0:
                clauses.append(clause)

        # Dataset generation
        solver = minisolvers.MinisatSolver()
        # Check SAT
        for i in range(no_var):
            solver.new_var(dvar=True)
        for c in clauses:
            solver.add_clause(c)

        if solver.solve(): 
            assignment = solver.get_model()
            x_data, edge_index = sat_utils.cnf_to_netlist(clauses, no_var)

            # Generate feature
            x_data, edge_data, level_list, fanin_list, fanout_list = circuit_utils.parse_x_data(x_data, edge_index)
            PI_list = level_list[0]
            if len(level_list[-1])!= 1:
                continue
            if len(PI_list) != no_var:
                continue
            
            # Generate x_data
            PO_idx = level_list[-1][0]
            x_data = circuit_utils.generate_prob_cont(x_data, PI_list, level_list, fanin_list)
            x_data = circuit_utils.generate_prob_obs(x_data, level_list, fanin_list, fanout_list)
            x_data, _ = circuit_utils.identify_reconvergence(x_data, level_list, fanin_list, fanout_list)
            mask_x_data = copy.deepcopy(x_data)
            for idx, x_data_info in enumerate(mask_x_data):
                mask_x_data[idx].append(-1)
            
            success, y = circuit_utils.search_solution(mask_x_data, PI_list, level_list, fanin_list, list(assignment))

            if success: 
                new_y_data = []
                for idx in range(len(y)):
                    new_y_data.append([y[idx]])
                return mask_x_data, edge_data, new_y_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='../dataset/kcolor/kcolor_n6-10_p0.37_c2-4', help='output directory')
    parser.add_argument('--N', type=int, default=100,help='number of problems to be generated')
    parser.add_argument('--min_n', type=int, default=6, help='lower bound of graph nodes')
    parser.add_argument('--max_n', type=int, default=10, help='upper bound of graph nodes')
    parser.add_argument('--p', type=float, default=0.37, help='probability of edge')
    parser.add_argument('--min_k', type=int, default=2, help='lower bound of number of colors')
    parser.add_argument('--max_k', type=int, default=4, help='upper bound of number of colors')
    parser.add_argument('--id', type=int, default=0, help='starting id')
    args = parser.parse_args()

    try:
        os.makedirs(args.dir)
    except OSError:
        if not os.path.isdir(args.dir):
            raise

    graphs = {}
    labels = {}
    for i in range(args.N):
        n = random.randint(args.min_n, args.max_n)
        k = random.randint(args.min_k, args.max_k)
        filename = 'id={}_n={}_p={}_k={}.cnf'.format(args.id + i, n, args.p, k)
        # x, edge, y = create_sat_problem(filename, n, args.p, k)
        x, edge, y = create_kcolor_problem(n, 0.37)
        graphs[filename] = {'x': np.array(x).astype('float32'), 'edge_index': np.array(edge)}
        labels[filename] = {'y': np.array(y).astype('float32')}

    circuits_file = args.dir + '/' + 'graphs.npz'
    labels_file = args.dir + '/' + 'labels.npz'
    np.savez_compressed(circuits_file, circuits=graphs)
    np.savez_compressed(labels_file, labels=labels)

if __name__ == '__main__':
    main()
