
for d in 1 2 3 4 5
do
    PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.01 \
    --emb_epochs 500 \
    --act tanh \
    --embedding_dim 90 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest > output/GAlign/ppi_del_edges_d${d}
done




for d in 1 
do
    PD=$HOME/dataspace/graph/douban
    PREFIX1=online
    PREFIX2=offline
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/${PREFIX1}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/dictionaries/groundtruth \
    --lr 0.01 \
    --neg_sample_size 5 \
    --emb_epochs 100 \
    --act tanh \
    --embedding_dim 75 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest > output/GAlign/douban
done

