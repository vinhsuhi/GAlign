home=$HOME

python utils/split_dict.py \
--input $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/random-a1-d1/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/random-a1-d1/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/random-a1-d1/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/random-a1-d1/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/random-a1-d1/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/random-a1-d1/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/random-a1-d1/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/random-a1-d1/dictionaries/ \
--split 0.2

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/random-a1-d1/dictionaries/node,split=0.2.test.dict \
IONE \
--gt_train $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/random-a1-d1/dictionaries/node,split=0.2.train.dict \
--total_iter 10000000 \
--dim 200 > log/ione/erdos-renyi-n1000-p1

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/random-a1-d1/dictionaries/node,split=0.2.test.dict \
IONE \
--gt_train $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/random-a1-d1/dictionaries/node,split=0.2.train.dict \
--total_iter 60000000 \
--dim 950 > log/ione/erdos-renyi-n10000-p05

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1/dictionaries/node,split=0.2.test.dict \
IONE \
--gt_train $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1/dictionaries/node,split=0.2.train.dict \
--total_iter 60000000 \
--dim 950 > log/ione/erdos-renyi-n10000-p1

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/random-a1-d1/dictionaries/node,split=0.2.test.dict \
IONE \
--gt_train $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/random-a1-d1/dictionaries/node,split=0.2.train.dict \
--total_iter 60000000 \
--dim 950 > log/ione/erdos-renyi-n10000-p2

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/random-a1-d1/dictionaries/node,split=0.2.test.dict \
IONE \
--gt_train $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/random-a1-d1/dictionaries/node,split=0.2.train.dict \
--total_iter 60000000 \
--dim 950 > log/ione/erdos-renyi-n10000-p3