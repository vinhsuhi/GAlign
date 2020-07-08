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
    NAWAL \
    --train_dict ${TRAIN} \
    --embedding_epochs 1000 \
    --test_dict ${TEST} > output/NAWAL/econ_del_edges_d${d}
done


for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/bn
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    NAWAL \
    --train_dict ${TRAIN} \
    --embedding_epochs 1000 \
    --test_dict ${TEST} > bn_del_edges_d${d}
done



for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    NAWAL \
    --train_dict ${TRAIN} \
    --embedding_epochs 1000 \
    --test_dict ${TEST} > ppi_del_edges_d${d}
done

for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/del-nodes-p${d}-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    NAWAL \
    --train_dict ${TRAIN} \
    --embedding_epochs 1000 \
    --test_dict ${TEST} > output/NAWAL/econ_del_nodes_d${d}
done


for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/bn
    PREFIX2=semi-synthetic/del-nodes-p${d}-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    NAWAL \
    --train_dict ${TRAIN} \
    --embedding_epochs 1000 \
    --test_dict ${TEST} > bn_del_nodes_d${d}
done



for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3
    PREFIX2=semi-synthetic/del-nodes-p${d}-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    NAWAL \
    --train_dict ${TRAIN} \
    --embedding_epochs 1000 \
    --test_dict ${TEST} > ppi_del_nodes_d${d}
done


