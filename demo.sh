
# Allmv_tmdb
python -u network_alignment.py --source_dataset ../dataspace/graph/suhi_allmv_tmdb/allmv/graphsage --target_dataset ../dataspace/graph/suhi_allmv_tmdb/tmdb/graphsage --groundtruth ../dataspace/graph/suhi_allmv_tmdb/dictionaries/groundtruth GAlign --log --GAlign_epochs 10 --refinement_epochs 50
# Douban
python -u network_alignment.py --source_dataset ../dataspace/graph/douban/online/graphsage --target_dataset ../dataspace/graph/douban/offline/graphsage --groundtruth ../dataspace/graph/douban/dictionaries/groundtruth GAlign --log --GAlign_epochs 50



python -u network_alignment.py --source_dataset ../dataspace/graph/douban/online/graphsage --target_dataset ../dataspace/graph/douban/offline/graphsage --groundtruth ../dataspace/graph/douban/dictionaries/node,split=0.2.test.dict IONE --train_dict ../dataspace/graph/douban/dictionaries/node,split=0.2.train.dict


for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    FINAL \
    --train_dict ${TRAIN} > output/FINAL/econ_del_edges_d${d}
done


python -u network_alignment.py --source_dataset ../dataspace/graph/econ-mahindas/graphsage --target_dataset ../dataspace/graph/econ-mahindas/semi-synthetic/REGAL-d1-seed1/graphsage --groundtruth ../dataspace/graph/econ-mahindas/semi-synthetic/REGAL-d1-seed1/dictionaries/node,split=0.2.test.dict FINAL --train_dict ../dataspace/graph/econ-mahindas/semi-synthetic/REGAL-d1-seed1/dictionaries/node,split=0.2.train.dict