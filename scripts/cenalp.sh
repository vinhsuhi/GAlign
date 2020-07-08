
PD=${HOME}/dataspace/graph/karate
PREFIX2=permutation
TRAINRATIO=0.2

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
CENALP \
--train_dict ${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--num_walks 20 \
--walk_len 5 \
--batch_size 512 \
--threshold 0.5 \
--linkpred_epochs 0 \
--num_pair_toadd 10 \
--num_sample 300 \
--cuda 

PD=${HOME}/dataspace/graph/karate
PREFIX2=del-nodes-p1-seed1
TRAINRATIO=0.1

python network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
CENALP \
--train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
--num_walks 20 \
--walk_len 5 \
--batch_size 512 \
--threshold 0.5 \
--linkpred_epochs 0 \
--num_pair_toadd 10 \
--num_sample 300 \
--cuda

# tuning with following notices:
# 1: First set the linkpred_epochs to 0, then tunning other hyper params to achieve the best result.
# 2: num_pair_toadd: Set int, higher value to faster training, but lower accuracy.
# 3: threshold, this is matter only if you use linkpred_epochs > 0, you should set it higher or equal 0.5
# 4: num_sample, this is matter only if you use linkpred_epochs > 0, you sould try 300, 400, ... 1000