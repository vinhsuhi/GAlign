################################ DEL EDGES ##########################################

# Supervised Models: FINAL, IsoRank, PALE (NAWAL), DeepLink
# Unsupervised Models: REGAL

for n in 1000 
do
    for m in 5 
    do

        X=p01-seed1
        PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3        
        PREFIX2=permut
        TRAINRATIO=0.2
        TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
        TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

        python -u network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX2}/graphsage/ \
        --groundtruth ${TEST} \
        --seed 111 \
        IsoRank 

done
done


for n in 1000 5000 10000
do
    for m in 5 10 20 30 50
    do

        X=p01-seed1
        PD=dataspace/dataspace/n${n}-m${m}
        PREFIX2=REGAL-d01-seed1
        TRAINRATIO=0.2
        TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
        TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

        python -u network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX2}/graphsage/ \
        --groundtruth ${TEST} \
        --seed 111 \
        BigAlign  > output/BigAlign/n${n}-m${m}_del_edges_d01

done
done


# REGAL
for n in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
    for m in 5 10 20 30 50
    do
        X=p01-seed1
        PD=dataspace/dataspace/n${n}-m${m}
        PREFIX2=REGAL-d01-seed1
        TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

        python -u network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX2}/graphsage/ \
        --groundtruth ${TEST} \
        --seed 111 \
        REGAL > output/REGAL/n${n}-m${m}_del_edges_d01

done
done

# BigAlign
for n in 1000 5000 10000 15000 20000 25000 30000 35000
do
    for m in 5 10 20 30 50
    do
        X=p01-seed1
        PD=dataspace/dataspace/n${n}-m${m}
        PREFIX2=REGAL-d01-seed1
        TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

        python -u network_alignment.py \
        --source_dataset ${PD}/graphsage/ \
        --target_dataset ${PD}/${PREFIX2}/graphsage/ \
        --groundtruth ${TEST} \
        --seed 111 \
        BigAlign > output/BigAlign/n${n}-m${m}_del_edges_d01
done
done
# IsoRank
for n in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
    for m in 5 10 20 30 50
    do
    X=p01-seed1
    PD=dataspace/dataspace/n${n}-m${m}
    PREFIX2=REGAL-d01-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    IsoRank \
    --alpha 0.001 > output/IsoRank/n${n}-m${m}_del_edges_d01

done
done

done


# DeepLink
for n in 5000 6000 7000 8000 9000 10000
do
    for m in 5 10 20 30 50
    do
    X=p01-seed1
    PD=dataspace/dataspace/n${n}-m${m}
    PREFIX2=REGAL-d01-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    DeepLink \
    --embedding_epochs 5 \
    --number_walks 20 \
    --unsupervised_epochs 500 \
    --supervised_epochs 500 \
    --walk_length 3 \
    --window_size 3 \
    --top_k 2 \
    --alpha 0.9 \
    --unsupervised_lr 0.01 \
    --supervised_lr 0.01 \
    --batch_size_mapping 50 \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --cuda > output/DeepLink/n${n}-m${m}_del_edges_d01
done
done

# PALE
for n in 6000 7000 8000 9000 10000
do
    for m in 5 10 20 30 50
    do
    X=p01-seed1
    PD=dataspace/dataspace/n${n}-m${m}
    PREFIX2=REGAL-d01-seed1
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
    --test_dict ${TEST} > output/PALE/n${n}-m${m}_del_edges_d01

done
done



# IONE
for n in 20000
do
    for m in 5 10 20 30 50
    do
    X=p01-seed1
    PD=dataspace/dataspace/n${n}-m${m}
    PREFIX2=REGAL-d01-seed1
    TRAINRATIO=0.2
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    IONE \
    --lr 0.1 \
    --epochs 150 \
    --train_dict ${TRAIN} > output/IONE/n${n}-m${m}_del_edges_d01 \

    rm -rf temp/*

done
done

for n in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
    for m in 5 10 20 30 50
    do

    python -m  fix_real_feats\
    --source_path dataspace/dataspace/n${n}-m${m}/graphsage \
    --target_path dataspace/dataspace/n${n}-m${m}/REGAL-d01-seed1/graphsage \
    --groundtruth_path dataspace/dataspace/n${n}-m${m}/REGAL-d01-seed1/dictionaries/groundtruth 

    done
done




REGAL FINAL IsoRank BigAlign PALE IONE DeepLink

REGAL 10^1.5 - 10^3.5: 10^2
FINAL same
IsoRank same
BigAlign 0.5 - 3
PALE 3 6
IONE same 
DeepLink same