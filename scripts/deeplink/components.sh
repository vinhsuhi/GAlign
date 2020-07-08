home=$HOME
fully=$home/dataspace/graph/fully-synthetic/connected-components


ALG=DeepLink
TRAINRATIO=0.2
NWALKS=50
WLEN=3
WSIZE=3
TOPK=5
AL=0.9
EMEP=5
SEP=2000
USEP=2000
# for nc in 1 2 3 4 5 6
# do
#     DATA=small-world-n1000-k10-p5-nc$nc

#     python utils/split_dict.py \
#         --input $fully/$DATA/random-d01/dictionaries/groundtruth \
#         --out_dir $fully/$DATA/random-d01/dictionaries/ \
#         --split 0.2


#     python -u network_alignment.py \
#     --source_dataset $fully/$DATA/graphsage \
#     --target_dataset $fully/$DATA/random-d01/graphsage \
#     --groundtruth $fully/$DATA/random-d01/dictionaries/node,split=0.2.test.dict \
#     $ALG \
#     --embedding_epochs ${EMEP} \
#     --unsupervised_epochs ${USEP} \
#     --supervised_epochs ${SEP} \
#     --train_dict $fully/$DATA/random-d01/dictionaries/node,split=0.2.train.dict \
#     --number_walks ${NWALKS} \
#     --walk_length ${WLEN} \
#     --window_size ${WSIZE} \
#     --top_k ${TOPK} \
#     --alpha ${AL} \
#     --fullgt $fully/$DATA/random-d01/dictionaries/node,split=0.2.train.dict \
#     --num_cores 8 \
#     --cuda > log/$ALG/$DATA
# done

python seeds.py --data_dir $fully \
--regex_dataname small-world-n1000-k10-p5-nc.-seed1.. \
--sub_dir_src_dataset graphsage/ \
--sub_dir_trg_dataset random-d01/graphsage \
--sub_dir_log components \
$ALG \
--embedding_epochs ${EMEP} \
--unsupervised_epochs ${USEP} \
--supervised_epochs ${SEP} \
--number_walks ${NWALKS} \
--walk_length ${WLEN} \
--window_size ${WSIZE} \
--top_k ${TOPK} \
--alpha ${AL} \
--fullgt $fully/$DATA/random-d01/dictionaries/node,split=0.2.train.dict \
--num_cores 8 \
--cuda
