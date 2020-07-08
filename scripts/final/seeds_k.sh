home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

python seeds.py --data_dir $fully \
--regex_dataname small-world-n1000-k[0-9]+-p5-seed1[01234]. \
--sub_dir_src_dataset graphsage/ \
--sub_dir_trg_dataset random-d01/graphsage \
--sub_dir_log density \
FINAL \
--max_iter 30 \
--alpha 0.82