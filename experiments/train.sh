# If enable gpu, please add '--gpus 0'

cd src
python main.py deepsat --exp_id deepsat_sr3to10_gru \
 --data_dir ../data/sr3_10 --dataset deepsat \
 --num_rounds 1 --batch_size 64 \
 --gate_types INPUT,AND,NOT --dim_node_feature 3 \
 --aggr_function agnnconv --update_function gru --wx_update --mask
