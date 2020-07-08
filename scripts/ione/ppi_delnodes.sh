home=$HOME
for i in 01 05 1 2 3 4 5
do
    python utils/split_dict.py \
    --input $home/dataspace/graph/ppi/subgraphs/subgraph3/del-nodes-p$i/dictionaries/groundtruth \
    --out_dir $home/dataspace/graph/ppi/subgraphs/subgraph3/del-nodes-p$i/dictionaries \
    --split 0.2

    python -u network_alignment.py --source_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/graphsage \
    --target_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/del-nodes-p$i/graphsage \
    --groundtruth $home/dataspace/graph/ppi/subgraphs/subgraph3/del-nodes-p$i/dictionaries/node,split=0.2.test.dict \
    IONE \
    --train_dict $home/dataspace/graph/ppi/subgraphs/subgraph3/del-nodes-p$i/dictionaries/node,split=0.2.train.dict \
    --total_iter 10000000 \
    --dim 200 > log/ione/ppi-delnodes-p$i
done
