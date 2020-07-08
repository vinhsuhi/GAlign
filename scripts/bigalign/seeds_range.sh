home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

python seeds.py --data_dir $fully \
--regex_dataname small-world-n1000-k10-p.-seed1[01]. \
--sub_dir_src_dataset graphsage/ \
--sub_dir_trg_dataset random-d01/graphsage \
BigAlign