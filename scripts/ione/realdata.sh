home=$HOME

python utils/split_dict.py \
--input $home/dataspace/graph/douban/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/douban/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/flickr_lastfm/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/flickr_lastfm/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/flickr_myspace/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/flickr_myspace/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/arenas/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/arenas/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/ppi/subgraphs/subgraph3/permut/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/ppi/subgraphs/subgraph3/permut/dictionaries \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fb-tw-data/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fb-tw-data/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fq-tw-data/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fq-tw-data/dictionaries/ \
--split 0.2


python -u network_alignment.py --source_dataset $home/dataspace/graph/douban/online/graphsage \
--target_dataset $home/dataspace/graph/douban/offline/graphsage \
--groundtruth $home/dataspace/graph/douban/dictionaries/node,split=0.2.test.dict \
IONE \
--train_dict $home/dataspace/graph/douban/dictionaries/node,split=0.2.train.dict \
--total_iter 10000000 \
--dim 200 > log/ione/douban

python -u network_alignment.py --source_dataset $home/dataspace/graph/flickr_lastfm/flickr/graphsage \
--target_dataset $home/dataspace/graph/flickr_lastfm/lastfm/graphsage \
--groundtruth $home/dataspace/graph/flickr_lastfm/dictionaries/node,split=0.2.test.dict \
IONE \
--train_dict $home/dataspace/graph/flickr_lastfm/dictionaries/node,split=0.2.train.dict \
--total_iter 40000000 \
--dim 950 > log/ione/flickr_lastfm

python -u network_alignment.py --source_dataset $home/dataspace/graph/flickr_myspace/flickr/graphsage \
--target_dataset $home/dataspace/graph/flickr_myspace/myspace/graphsage \
--groundtruth $home/dataspace/graph/flickr_myspace/dictionaries/node,split=0.2.test.dict \
IONE \
--train_dict $home/dataspace/graph/flickr_myspace/dictionaries/node,split=0.2.train.dict \
--total_iter 10000000 \
--dim 100 > log/ione/flickr_myspace

python -u network_alignment.py --source_dataset $home/dataspace/graph/arenas/arenas1/graphsage \
--target_dataset $home/dataspace/graph/arenas/arenas2/graphsage \
--groundtruth $home/dataspace/graph/arenas/dictionaries/node,split=0.2.test.dict \
IONE \
--train_dict $home/dataspace/graph/arenas/dictionaries/node,split=0.2.train.dict > log/ione/arenas

python -u network_alignment.py --source_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/graphsage \
--target_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/permut/graphsage \
--groundtruth $home/dataspace/graph/ppi/subgraphs/subgraph3/permut/dictionaries/node,split=0.2.test.dict \
IONE \
--train_dict $home/dataspace/graph/ppi/subgraphs/subgraph3/permut/dictionaries/node,split=0.2.train.dict \
--total_iter 10000000 \
--dim 200 > log/ione/ppi

python -u network_alignment.py --source_dataset $home/dataspace/graph/fb-tw-data/facebook/graphsage \
--target_dataset $home/dataspace/graph/fb-tw-data/twitter/graphsage \
--groundtruth $home/dataspace/graph/fb-tw-data/dictionaries/node,split=0.2.test.dict \
IONE \
--train_dict $home/dataspace/graph/fb-tw-data/dictionaries/node,split=0.2.train.dict \
--total_iter 80000000 \
--dim 1300 > log/ione/fb-tw-data

python -u network_alignment.py --source_dataset $home/dataspace/graph/fq-tw-data/foursquare/graphsage \
--target_dataset $home/dataspace/graph/fq-tw-data/twitter/graphsage \
--groundtruth $home/dataspace/graph/fq-tw-data/dictionaries/node,split=0.2.test.dict \
IONE \
--train_dict $home/dataspace/graph/fq-tw-data/dictionaries/node,split=0.2.train.dict \
--total_iter 80000000 \
--dim 1300 > log/ione/fq-tw-data
