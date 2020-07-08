PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3
PREFIX2=semi-synthetic/REGAL-d2-seed1

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
GAlign \
--embedding_dim 200 \
--emb_epochs 20 \
--lr 0.01 \
--num_MSA_blocks 2 \
--noise_level 0.01 \
--log \
--refine \
--cuda 
