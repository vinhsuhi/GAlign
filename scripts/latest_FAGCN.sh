PD=$HOME/dataspace/graph/econ-mahindas
PREFIX2=semi-synthetic/REGAL-d2-seed1

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
GAlign \
--embedding_dim 200 \
--emb_epochs 100 \
--lr 0.01 \
--num_GCN_blocks 2 \
--noise_level 0 \
--source_embedding ${PD}/graphsage/mincut_emb.npy \
--target_embedding ${PD}/${PREFIX2}/graphsage/mincut_emb.npy \
--train_dict ${PD}/${PREFIX2}/dictionaries/node,split=0.2.train.dict \
--log \
--cuda 


PD=$HOME/dataspace/graph/econ-mahindas
PREFIX2=semi-synthetic/REGAL-d2-seed1

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
GAlign \
--embedding_dim 200 \
--emb_epochs 100 \
--lr 0.01 \
--num_GCN_blocks 2 \
--noise_level 0 \
--log 



PD=$HOME/dataspace/graph/douban

python -u network_alignment.py \
--source_dataset ${PD}/online/graphsage/ \
--target_dataset ${PD}/offline/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
GAlign \
--embedding_dim 200 \
--emb_epochs 100 \
--lr 0.01 \
--num_GCN_blocks 2 \
--noise_level 0 \
--source_embedding ${PD}/graphsage/mincut_emb.npy \
--train_dict ${PD}/${PREFIX2}/dictionaries/node,split=0.2.train.dict \
--log 



PD=$HOME/dataspace/graph/douban/
PREFIX2=semi-synthetic/REGAL-d2-seed1

python -u network_alignment.py \
--source_dataset ${PD}/online/graphsage/ \
--target_dataset ${PD}/offline/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
GAlign \
--embedding_dim 200 \
--emb_epochs 100 \
--lr 0.01 \
--num_GCN_blocks 2 \
--noise_level 0 \
--log \
--cuda 


python -u network_alignment.py `
--source_dataset ../networkAlignment/data/fully-synthetic/erdos-renyi-n30-p3/graphsage/ `
--target_dataset ../networkAlignment/data/fully-synthetic/erdos-renyi-n30-p3/permut/graphsage/ `
--groundtruth ../networkAlignment/data/fully-synthetic/erdos-renyi-n30-p3/permut/dictionaries/groundtruth `
GAlign `
--embedding_dim 200 `
--emb_epochs 10 `
--lr 0.01 `
--num_GCN_blocks 2 `
--noise_level 0 `
--log `
--refine


python -m utils.extract_data_from_graph `
--source_dataset ../dataspace/graph/flickr_myspace/flickr/graphsage/ `
--target_dataset ../dataspace/graph/flickr_myspace/myspace/graphsage/ `
--groundtruth ../dataspace/graph/flickr_myspace/dictionaries/groundtruth `
