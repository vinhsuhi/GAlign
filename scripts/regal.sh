
for X in "d01-seed1" "d05-seed1" "d1-seed1" "d2-seed1" "d3-seed1" "d4-seed1" "d5-seed1"
do
	PD=$HOME/dataspace/graph/pale_facebook
	PREFIX2=REGAL-${X}

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    REGAL > output/REGAL/${PREFIX2}_facebook

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    FINAL > output/FINAL/${PREFIX2}_facebook
done



for X in "d01-seed1" "d05-seed1" "d1-seed1" "d2-seed1" "d3-seed1" "d4-seed1" "d5-seed1"
do
	PD=$HOME/dataspace/graph/fb-tw-data/twitter
	PREFIX2=REGAL-${X}

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    REGAL > output/REGAL/${PREFIX2}_twitter

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    FINAL > output/FINAL/${PREFIX2}_twitter
done


for X in "d01-seed1" "d05-seed1" "d1-seed1" "d2-seed1" "d3-seed1" "d4-seed1" "d5-seed1"
do
	PD=$HOME/dataspace/graph/fq-tw-data/foursquare
	PREFIX2=REGAL-${X}

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    REGAL > output/REGAL/${PREFIX2}_foursquare

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    FINAL > output/FINAL/${PREFIX2}_foursquare
done


PD=$HOME/dataspace/graph/pale_facebook
PREFIX2=permut

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
REGAL > output/REGAL/${PREFIX2}_facebook

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
FINAL > output/FINAL/${PREFIX2}_facebook


PD=$HOME/dataspace/graph/fb-tw-data/twitter
PREFIX2=permut

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
REGAL > output/REGAL/${PREFIX2}_twitter

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
FINAL > output/FINAL/${PREFIX2}_twitter


PD=$HOME/dataspace/graph/ppi
PREFIX2=permut

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
REGAL 

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
FINAL > output/FINAL/${PREFIX2}_ppi



for X in 01 05 1 2 3 4 5
do
	PD=$HOME/dataspace/graph/pale_facebook
	PREFIX2=del-nodes-p${X}-seed1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    REGAL > output/REGAL/${PREFIX2}_facebook

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    FINAL > output/FINAL/${PREFIX2}_facebook
done



for X in 01 05 1 2 3 4 5
do
	PD=$HOME/dataspace/graph/fb-tw-data/twitter
	PREFIX2=del-nodes-p${X}-seed1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    REGAL > output/REGAL/${PREFIX2}_twitter

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    FINAL > output/FINAL/${PREFIX2}_twitter
done


for X in 01 05 1 2 3 4 5
do
	PD=$HOME/dataspace/graph/fq-tw-data/foursquare
	PREFIX2=del-nodes-p${X}-seed1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    REGAL > output/REGAL/${PREFIX2}_foursquare

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    FINAL > output/FINAL/${PREFIX2}_foursquare
done




for X in 01 05 1 2 3 4 5
do
	PD=$HOME/dataspace/graph/ppi
	PREFIX2=del-nodes-p${X}-seed1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    REGAL > output/REGAL/${PREFIX2}_ppi

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    FINAL > output/FINAL/${PREFIX2}_ppi
done


PD=$HOME/dataspace/graph/fb-tw-data/twitter
PREFIX2=permut
TRAINRATIO=0.1

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
DeepLink \
--train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--cuda > output/DeepLink/${PREFIX2}_twitter


PD=$HOME/dataspace/graph/pale_facebook
PREFIX2=permut
TRAINRATIO=0.1

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
DeepLink \
--train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--cuda > output/DeepLink/${PREFIX2}_facebook



PD=$HOME/dataspace/graph/fq-tw-data/foursquare
PREFIX2=permut
TRAINRATIO=0.1

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
DeepLink \
--train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--cuda > output/DeepLink/${PREFIX2}_foursquare



for X in 01 05 1 2 3 4 5
do
    PD=$HOME/dataspace/graph/ppi
    PREFIX2=del-nodes-p${X}-seed1
    TRAINRATIO=0.1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    DeepLink \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --cuda > output/DeepLink/${PREFIX2}_ppi
done




for X in 01 05 1 2 3 4 5
do
    PD=$HOME/dataspace/graph/pale_facebook
    PREFIX2=del-nodes-p${X}-seed1
    TRAINRATIO=0.1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    DeepLink \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --cuda > output/DeepLink/${PREFIX2}_facebook
done



for X in 01 05 1 2 3 4 5
do
    PD=$HOME/dataspace/graph/fb-tw-data/twitter
    PREFIX2=del-nodes-p${X}-seed1
    TRAINRATIO=0.1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    DeepLink \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --cuda > output/DeepLink/${PREFIX2}_twitter
done


for X in 01 05 1 2 3 4 5
do
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=del-nodes-p${X}-seed1
    TRAINRATIO=0.1

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    DeepLink \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --cuda > output/DeepLink/${PREFIX2}_foursquare
done

