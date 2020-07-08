home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

python seeds.py --data_dir $fully \
--regex_dataname small-world-n1000-k..-p5-seed1[01234]. \
--sub_dir_src_dataset graphsage/ \
--sub_dir_trg_dataset random-d01/graphsage \
--sub_dir_log density \
IONE \
--total_iter 10000000 \
--dim 200