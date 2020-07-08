home=$HOME
ppi=$home/dataspace/graph/ppi/subgraphs/subgraph3
for i in `ls $ppi/semi-synthetic | grep pfeats`
do
    python utils/split_dict.py \
    --input $ppi/semi-synthetic/$i/dictionaries/groundtruth \
    --out_dir $ppi/semi-synthetic/$i/dictionaries/ \
    --split 0.2

    python -u network_alignment.py \
    --source_dataset $ppi/graphsage \
    --target_dataset $ppi/semi-synthetic/$i/graphsage \
    --groundtruth $ppi/semi-synthetic/$i/dictionaries/node,split=0.2.test.dict \
    IONE \
    --train_dict $ppi/semi-synthetic/$i/dictionaries/node,split=0.2.train.dict \
    --total_iter 10000000 \
    --dim 200 > log/ione/$i
done
