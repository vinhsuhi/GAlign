############# ISORANK ##############
home=$HOME
python -u network_alignment.py --source_dataset $home/dataspace/graph/douban/online/graphsage \
--target_dataset $home/dataspace/graph/douban/offline/graphsage \
--groundtruth $home/dataspace/graph/douban/dictionaries/groundtruth \
IsoRank \
--H $home/dataspace/graph/douban/H.mat \
--max_iter 30 \
--alpha 0.82 > log/isorank/douban

python -u network_alignment.py --source_dataset $home/dataspace/graph/flickr_lastfm/flickr/graphsage \
--target_dataset $home/dataspace/graph/flickr_lastfm/lastfm/graphsage \
--groundtruth $home/dataspace/graph/flickr_lastfm/dictionaries/groundtruth \
IsoRank \
--H $home/dataspace/graph/flickr_lastfm/H.mat \
--max_iter 30 \
--alpha 0.3 > log/isorank/flickr_lastfm

python -u network_alignment.py --source_dataset $home/dataspace/graph/flickr_myspace/flickr/graphsage \
--target_dataset $home/dataspace/graph/flickr_myspace/myspace/graphsage \
--groundtruth $home/dataspace/graph/flickr_myspace/dictionaries/groundtruth \
IsoRank \
--H $home/dataspace/graph/flickr_myspace/H.mat \
--max_iter 30 \
--alpha 0.3 > log/isorank/flickr_myspace

python -u network_alignment.py --source_dataset $home/dataspace/graph/arenas/arenas1/graphsage \
--target_dataset $home/dataspace/graph/arenas/arenas2/graphsage \
--groundtruth $home/dataspace/graph/arenas/dictionaries/groundtruth \
IsoRank \
--max_iter 30 \
--alpha 0.82 > log/isorank/arenas

python -u network_alignment.py --source_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/graphsage \
--target_dataset $home/dataspace/graph/ppi/subgraphs/subgraph3/permut/graphsage \
--groundtruth $home/dataspace/graph/ppi/subgraphs/subgraph3/permut/dictionaries/groundtruth \
IsoRank \
--max_iter 30 \
--alpha 0.82 > log/isorank/ppi

python -u network_alignment.py --source_dataset $home/dataspace/graph/fb-tw-data/facebook/graphsage \
--target_dataset $home/dataspace/graph/fb-tw-data/twitter/graphsage \
--groundtruth $home/dataspace/graph/fb-tw-data/dictionaries/groundtruth \
IsoRank \
--max_iter 30 \
--alpha 0.82 > log/isorank/fb-tw-data

python -u network_alignment.py --source_dataset $home/dataspace/graph/fq-tw-data/foursquare/graphsage \
--target_dataset $home/dataspace/graph/fq-tw-data/twitter/graphsage \
--groundtruth $home/dataspace/graph/fq-tw-data/dictionaries/groundtruth \
IsoRank \
--max_iter 30 \
--alpha 0.82 > log/isorank/fq-tw-data
