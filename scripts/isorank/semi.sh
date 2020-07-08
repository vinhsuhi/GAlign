home=$HOME
for i in 001 005 01 05 1 2
do
    python -u network_alignment.py \
    --source_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/graphsage \
    --target_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/REGAL-d$i/graphsage \
    --groundtruth $home/dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/REGAL-d$i/dictionaries/groundtruth \
    IsoRank \
    --max_iter 30 \
    --alpha 0.82 > log/isorank/ppi-regal-d$i
done
