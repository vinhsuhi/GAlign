PD=$HOME/dataspace/graph/douban/

python -u network_alignment.py \
--source_dataset ${PD}/online/graphsage/ \
--target_dataset ${PD}/offline/graphsage/ \
--groundtruth ${PD}/dictionaries/groundtruth \
GAlign \
--embedding_dim 200 \
--emb_epochs 20 \
--lr 0.01 \
--num_MSA_blocks 2 \
--noise_level 0.001 \
--log \
--refine


python -u network_alignment.py --source_dataset ../dataspace/graph/douban/online/graphsage --target_dataset ../dataspace/graph/douban/offline/graphsage --groundtruth ../dataspace/graph/douban/dictionaries/groundtruth GAlign --log --GAlign_epochs 50

python -u network_alignment.py --source_dataset ../dataspace/graph/suhi_allmv_tmdb/allmv/graphsage --target_dataset ../dataspace/graph/suhi_allmv_tmdb/tmdb/graphsage --groundtruth ../dataspace/graph/suhi_allmv_tmdb/dictionaries/groundtruth GAlign --log --GAlign_epochs 50
# PD=$HOME/dataspace/graph/econ-mahindas
# PREFIX2=semi-synthetic/REGAL-d2-seed1

# python -u network_alignment.py \
# --source_dataset ${PD}/graphsage/ \
# --target_dataset ${PD}/${PREFIX2}/graphsage/ \
# --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
# GAlign \
# --embedding_dim 200 \
# --emb_epochs 20 \
# --lr 0.01 \
# --num_MSA_blocks 2 \
# --noise_level 0.001 \
# --log

#Accuracy: 0.4311
#MAP: 0.5408
#Top_5: 0.6699
#Top_10: 0.7710
#Full_time: 11.9626
