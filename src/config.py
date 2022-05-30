import os
import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='Pytorch training script of DeepGate.')

    # basic experiment setting
    parser.add_argument('task', default='prob', choices=['deepsat'],
                             help='NIPS22 Supplementary Version current support: deepsat')
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--debug', type=int, default=0,
                             help='level of visualization.'
                                  '1: only show the final results compared with C1'
                                  '2: debug the network gradients')
                                #   '3: use matplot to display' # useful when lunching training with ipython notebook
                                #   '4: save all visualizations to disk')
    parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.')

    # system
    parser.add_argument('--gpus', default='-1', 
                             help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    parser.add_argument('--random-seed', type=int, default=208, 
                             help='random seed')

    # log
    parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    parser.add_argument('--save_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')


    # dataset settings
    parser.add_argument('--dataset', default='deepsat', type=str, choices=['deepsat'],
                             metavar='NAME', help='target dataset')
    parser.add_argument('--data_dir', default='../data/random_circuits',
                             type=str, help='the path to the dataset')
    parser.add_argument('--test_data_dir', default=None,
                             type=str, help='the path to the testing dataset')
    # circuit
    parser.add_argument('--gate_types', default='*', type=str,
                             metavar='LIST', help='gate types in the circuits. For aig: INPUT,AND,NOT, For Circuit-sat: INPUT,AND,OR,NOT')
    parser.add_argument('--dim_node_feature', default=8, 
                             type=int, help='the dimension of node features')
    parser.add_argument('--dim_edge_feature', default=16,
                             type=int, help='the dimension of node features')
    parser.add_argument('--small_train', default=0, 
                             type=int,help='if True, use a smaller version of train set')
    parser.add_argument('--un_directed', default=False, action='store_true', 
                             help='If true, model the circuit as the undirected graph. Default: circuit as DAG')
    # sat
    parser.add_argument('--n_pairs', default=10000, type=int, 
                             help='number of sat/unsat problems to generate')
    parser.add_argument('--min_n', type=int, default=3, 
                             help='min number of variables used for training')
    parser.add_argument('--max_n', type=int, default=10, 
                             help='max number of variables used for training')
    # neurosat
    parser.add_argument('--p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', type=float, default=0.4)
    # circuitsat
    parser.add_argument('--exp_depth', type=int, default=3)
    # deepsat
    parser.add_argument('--use_aig', action='store_true', 
                             help='whether to use AIG.')
    




    # model settings
    parser.add_argument('--arch', default='deepsat', choices=['deepsat'],
                             help='model architecture. NIPS22 Supplementary Version currently support'
                                  'deepsat')
    parser.add_argument('--activation_layer', default='relu', type=str, choices=['relu', 'relu6', 'sigmoid'],
                             help='The activation function to use in the FC layers.')  
    parser.add_argument('--norm_layer', default='batchnorm', type=str,
                             help='The normalization function to use in the FC layers.')
    parser.add_argument('--num_fc', default=3, type=int,
                             help='The number of FC layers')                          
    # recgnn / neurosat
    parser.add_argument('--num_aggr', default=3, type=int,
                             help='the number of aggregation layers.')
    parser.add_argument('--aggr_function', default='deepset', type=str, choices=['deepset', 'agnnconv', 'gated_sum', 'conv_sum'],
                             help='the aggregation function to use.')
    parser.add_argument('--update_function', default='gru', type=str, choices=['gru', 'lstm', 'layernorm_lstm', 'layernorm_gru'],
                             help='the update function to use.')
    parser.add_argument('--wx_update', action='store_true', default=False,
                            help='The inputs for the update function considers the node feature of mlp.')
    parser.add_argument('--no_keep_input', action='store_true', default=False,
                             help='no to use the input feature as the input to recurrent function.')
    parser.add_argument('--aggr_state', action='store_true', default=False,
                             help='use the aggregated message as the previous state of recurrent function.')
    parser.add_argument('--init_hidden', action='store_true', 
                             default=False, help='whether to init the hidden state of node embeddings')
    parser.add_argument('--num_rounds', type=int, default=1, metavar='N',
                             help='The number of rounds for grn propagation.'
                             '1 - the setting used in DAGNN/D-VAE')
    parser.add_argument('--no_reverse', action='store_true', default=False,
                             help='Not to use the reverse layer to propagate the message.')
    parser.add_argument('--seperate_hidden', action='store_true', default=False,
                             help='seperate node hidden states for forward layer and backward layer.')
    parser.add_argument('--dim_hidden', type=int, default=64, metavar='N',
                             help='hidden size of recurrent unit.')
    parser.add_argument('--dim_mlp', type=int, default=32, metavar='N',
                             help='hidden size of readout layers') 
    parser.add_argument('--dim_pred', type=int, default=1, metavar='N',
                             help='hidden size of readout layers')
    parser.add_argument('--mul_mlp', action='store_true', default=False,
                             help='To use seperate MLP for different gate types.') 
    parser.add_argument('--wx_mlp', action='store_true', default=False,
                             help='The inputs for the mlp considers the node feature of mlp.')        
    # deepsat
    parser.add_argument('--mask', action='store_true', default=False,
                             help='Use the mask for the node embedding or not')

    # circuitsat/deepsat
    parser.add_argument('--temperature', type=float, default=0.01,
                             help='initial value for temperature')
    parser.add_argument('--eplison', type=float, default=0.4,
                             help='the anneling factore of temperature.')
    parser.add_argument('--k_step', type=float, default=10.0,
                             help='the value for step funtion parameter k.')
    parser.add_argument('--prob_loss', action='store_true', default=False,
                             help='To use the simulated probabilities as complementary supervision.')
    parser.add_argument('--prob_weight', type=float, default=0.1,
                             help='the weight for simulated probability loss.')                       
             
    parser.add_argument('--hs_can_train', action='store_true', default=False)
    parser.add_argument('--nodag', action='store_true', default=False)
    parser.add_argument('--nolevel', action='store_true', default=False)


    # loss
    parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2 | focalloss')
    parser.add_argument('--cls_loss', default='bce_logit',
                             help='classification loss: bce - BCELoss | bce_logit - BCELossWithLogit | cross - CrossEntropyLoss')
    parser.add_argument('--sat_loss', default='smoothstep', choices=['smoothstep'])


    # train and val
    parser.add_argument('--lr', type=float, default=1.0e-4, 
                             help='learning rate for batch size 32.')
    parser.add_argument('--weight_decay', type=float, default=1e-10, 
                             help='weight decay (default: 1e-10)')
    parser.add_argument('--lr_step', type=str, default='30,45',
                             help='drop learning rate by 10.')
    parser.add_argument('--grad_clip', type=float, default=0.,
                        help='gradiant clipping')
    parser.add_argument('--num_epochs', type=int, default=60,
                             help='total training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    parser.add_argument('--master_batch_size', type=int, default=-1,
                             help='batch size on the master gpu.')
    parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    parser.add_argument('--trainval_split', default=0.9, type=float,
                             help='the splitting setting for training dataset and validation dataset.')
    parser.add_argument('--val_only', action='store_true', 
                             help='Do the validation evaluation only.')
    
    # test
    parser.add_argument('--test_split', default='test', choices=['test', 'train', 'all'],
                             help='the split to use for testing.')
    parser.add_argument('--cop_only', action='store_true',
                             help='only show the comparision between C1 and simluated probability.')
    parser.add_argument('--test_num_rounds', default=10, type=int,
                             help='The number of rounds to be run during testing.')
    
    args = parser.parse_args()

    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]
    args.lr_step = [int(i) for i in args.lr_step.split(',')]


    # update data settings
    DEFAULT_GATE_TO_INDEX = {"INPUT": 0, "AND": 1, "NAND": 2, "OR": 3, "NOR": 4, "NOT": 5, "XOR": 6}
    args.gate_to_index = {}
    if args.gate_types == '*':
        args.gate_to_index = DEFAULT_GATE_TO_INDEX
    else:
        gate_types = args.gate_types.split(',')
        for i in range(len(gate_types)):
            args.gate_to_index[gate_types[i]] = i
    args.num_gate_types = len(args.gate_to_index)

    # check the relationship of `task`, `dataset` and `arch` comply with each other. TODO: optimize this part
    if args.task == 'deepsat':
        assert args.dataset == 'deepsat', 'The dataset should be deepsat, if the task is deepsat.'
        assert args.arch == 'deepsat', 'The architecture should be deepsat, if the task is either deepsat.'
    else:
        raise
   
    
    if args.dataset == 'deepsat':
        args.circuit_file = "sat_circuits_graphs.npz"
        args.label_file = "sat_circuits_labels.npz"

    args.reverse = not args.no_reverse


    assert args.dim_node_feature == (len(args.gate_to_index)), "The dimension of node feature is not consistent with the specification, please check it again." 


    if args.debug > 0:
        args.num_workers = 0
        args.batch_size = 1
        args.gpus = [args.gpus[0]]


    # dir
    args.root_dir = os.path.join(os.path.dirname(__file__), '..')
    args.exp_dir = os.path.join(args.root_dir, 'exp', args.task)
    args.save_dir = os.path.join(args.exp_dir, args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    print('The output will be saved to ', args.save_dir)

    if args.resume and args.load_model == '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, 'model_last.pth')
    elif args.load_model != '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, args.load_model)


    # TODO List
    if args.mul_mlp: raise NotImplementedError   
    if args.debug == 2: raise NotImplementedError
    if len(args.gpus) > 1: raise NotImplementedError('Only support single gpu right now.')
    if args.seperate_hidden: raise NotImplementedError
    if args.reg_loss == 'focalloss': raise NotImplementedError
    


    return args
