home=$HOME
python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n1000-p2/random-a1-d1/dictionaries/groundtruth \
FINAL \
--max_iter 30 \
--alpha 0.82 > log/FINAL/erdos-renyi-n1000-p2

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p05/random-a1-d1/dictionaries/groundtruth \
FINAL \
--max_iter 30 \
--alpha 0.82 > log/FINAL/erdos-renyi-n10000-p05

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p1/random-a1-d1/dictionaries/groundtruth \
FINAL \
--max_iter 30 \
--alpha 0.82 > log/FINAL/erdos-renyi-n10000-p1

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p2/random-a1-d1/dictionaries/groundtruth \
FINAL \
--max_iter 30 \
--alpha 0.82 > log/FINAL/erdos-renyi-n10000-p2

python -u network_alignment.py \
--source_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/graphsage \
--target_dataset $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/random-a1-d1/graphsage \
--groundtruth $home/dataspace/graph/fully-synthetic/erdos-renyi-n10000-p3/random-a1-d1/dictionaries/groundtruth \
FINAL \
--max_iter 30 \
--alpha 0.82 > log/FINAL/erdos-renyi-n10000-p3