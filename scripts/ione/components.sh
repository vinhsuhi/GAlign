home=$HOME
fully=$home/dataspace/graph/fully-synthetic/connected-components

ALG=IONE
# for nc in 1 2 3 4 5 6
# do
#     DATA=small-world-n1000-k10-p5-nc$nc

#     python utils/split_dict.py \
#         --input $fully/$DATA/random-d01/dictionaries/groundtruth \
#         --out_dir $fully/$DATA/random-d01/dictionaries/ \
#         --split 0.2

#     python -u network_alignment.py \
#         --source_dataset $fully/$DATA/graphsage \
#         --target_dataset $fully/$DATA/random-d01/graphsage \
#         --groundtruth $fully/$DATA/random-d01/dictionaries/node,split=0.2.test.dict \
#         $ALG \
#         --train_dict $fully/$DATA/random-d01/dictionaries/node,split=0.2.train.dict \
#         --epochs 1000 \
#         --dim 200> log/$ALG/$DATA
# done
python seeds.py --data_dir $fully \
--regex_dataname small-world-n1000-k10-p5-nc.-seed1.. \
--sub_dir_src_dataset graphsage/ \
--sub_dir_trg_dataset random-d01/graphsage \
--sub_dir_log components \
$ALG \
--epochs 1000 \
--dim 200
