for d in 0 01 05 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-wt-data/facebook
    PREFIX2=REGAL-d${d}-seed1
    TRAINRATIO=0.1
    TRAIN=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TEST=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TEST} \
    --seed 111 \
    IONE \
    --train_dict ${TRAIN} > output/IONE/facebook_del_edges_d${d}
done



# IsoRank
for d in 0 01 05 1 2 3 4 5
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-wt-data/facebook
    PREFIX2=REGAL-d${d}-seed1
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
    --alpha 0.001 > output/IsoRank/facebook_del_edges_d${d}
done


# IONE phai chay tren bigdata moi duoc.
# NOTE: PREFIX2 khong co semi-synthetic
# FACEBOOK: PD=$HOME/dataspace/graph/fb-wt-data/facebook
# Twitter: PD=$HOME/dataspace/graph/fb-wt-data/twitter
# Foursquare PD=$HOME/dataspace/graph/fq-wt-data/foursquare
# del_node: PREFIX2=$del-nodes-p${d}-seed1
# noise_range: 0 01 05 1 2 3 4 5