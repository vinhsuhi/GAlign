home=$HOME
ppi=$home/dataspace/graph/ppi/subgraphs/subgraph3
for i in `ls $ppi/semi-synthetic | grep pfeats`
do
    python -u network_alignment.py \
    --source_dataset $ppi/graphsage \
    --target_dataset $ppi/semi-synthetic/$i/graphsage \
    --groundtruth $ppi/semi-synthetic/$i/dictionaries/groundtruth \
    BigAlign > log/bigalign/$i
done
