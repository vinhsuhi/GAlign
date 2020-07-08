home=$HOME
for i in 01 05 1 2 3 4 5
do
    python -u network_alignment.py --source_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/graphsage \
    --target_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/del-nodes-p$i/graphsage \
    --groundtruth $home/dataspace/graph/ppi/subgraphs/subgraph3/del-nodes-p$i/dictionaries/groundtruth \
    FINAL \
    --max_iter 30 \
    --alpha 0.82 > log/FINAL/ppi-delnodes-p$i
done
