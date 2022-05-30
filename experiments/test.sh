EXP_ID='deepsat_sr3to10_gru' # ['deepsat_sr3to10_gru']
ARRG='agnnconv' # ['deepset', 'agnnconv', 'gated_sum', 'conv_sum']
UPDATE='gru' # ['gru', 'lstm', 'layernorm_lstm', 'layernorm_gru']
ARCH='deepsat' 
TEST='sr10'
MODEL='model_trained.pth' 
cd src
python test.py deepsat --exp_id $EXP_ID \
 --data_dir ../data/$TEST --dataset deepsat \
 --num_rounds 1 --gpus -1 --batch_size 1 \
 --gate_types INPUT,AND,NOT --dim_node_feature 3 --wx_update \
 --aggr_function $ARRG --update_function $UPDATE --arch $ARCH --load_model $MODEL --test_num_rounds 1 --mask 
