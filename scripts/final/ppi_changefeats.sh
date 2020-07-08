home=$HOME
ppi=$home/dataspace/graph/ppi/subgraphs/subgraph3
for i in `ls $ppi/semi-synthetic | grep pfeats`
do
    python -u network_alignment.py \
    --source_dataset $ppi/graphsage \
    --target_dataset $ppi/semi-synthetic/$i/graphsage \
    --groundtruth $ppi/semi-synthetic/$i/dictionaries/groundtruth \
    FINAL \
    --max_iter 30 \
    --alpha 0.82 > log/FINAL/$i
done
