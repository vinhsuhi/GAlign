################################ DEL EDGES ##########################################

# Supervised Models: FINAL, IsoRank, PALE (NAWAL), DeepLink
# Unsupervised Models: REGAL

# FINAL
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

# REGAL
for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    REGAL > output/REGAL/econ_del_edges_d${d}
done

# IsoRank
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
    IsoRank \
    --train_dict ${TRAIN} \
    --alpha 0.001 > output/IsoRank/econ_del_edges_d${d}
done


# DeepLink
for d in 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    DeepLink \
    --embedding_epochs 5 \
    --number_walks 50 \
    --unsupervised_epochs 2000 \
    --supervised_epochs 2000 \
    --walk_length 3 \
    --window_size 3 \
    --top_k 2 \
    --alpha 0.9 \
    --unsupervised_lr 0.01 \
    --supervised_lr 0.01 \
    --batch_size_mapping 50 \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --cuda > output/DeepLink/econ_del_edges_d${d}
done

# PALE
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
    --test_dict ${TEST} > output/PALE/econ_del_edges_d${d}
done




# DEEPLINK
for d in 1
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    DeepLink \
    --embedding_epochs 5 \
    --number_walks 50 \
    --unsupervised_epochs 2000 \
    --supervised_epochs 2000 \
    --walk_length 3 \
    --window_size 3 \
    --top_k 2 \
    --alpha 0.9 \
    --unsupervised_lr 0.01 \
    --supervised_lr 0.01 \
    --batch_size_mapping 50 \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --cuda > output/DeepLink/econ_del_edges_d${d}
done

######################################################## DEL NODES ###############################

# FINAL
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
    FINAL \
    --train_dict ${TRAIN} > output/FINAL/econ_del_nodes_d${d}
done


################################## REAL DATASET #####################
# NOTEEEEEEEEE: FINAL for douban is special, don't use train_dict flag

# FINAL
for d in 1 
do
    ED=100
    PD=$HOME/dataspace/graph/douban
    PREFIX1=online
    PREFIX2=offline
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/${PREFIX1}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/dictionaries/groundtruth \
    --seed 111 \
    FINAL \
    --H ${PD}/H.mat > output/FINAL/douban
done



######################## TWITTER Here #############################

