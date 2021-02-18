
fully=$HOME/dataspace/graph/ppi/subgraphs/subgraph3

for cw in 0
do 
for num_walk in 10
do 
for walk_lenght in 3 
do 
python -u network_alignment.py \
--source_dataset $fully/graphsage \
--target_dataset $fully/del-nodes-p3-seed1/graphsage \
--groundtruth $fully/del-nodes-p3-seed1/dictionaries/node,split=0.2.test.dict \
PALE \
--embedding_epochs 100 \
--mapping_epochs 100 \
--train_dict $fully/del-nodes-p3-seed1/dictionaries/node,split=0.2.test.dict \
--batch_size_embedding 512 \
--cur_weight ${cw} \
--walk_len ${walk_lenght} \
--num_walks ${num_walk} \
--learning_rate1 0.001 \
--toy \
--cuda 
done
done 
done 

