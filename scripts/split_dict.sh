home=$HOME

##############fully#######
for i in 1, 2, 4, 6, 8, 10
do
  L=k$i0-p5

  python utils/split_dict.py --input $HOME/dataspace/graph/fully-synthetic/small-world-n1000-${L}/random-d01/dictionaries/groundtruth \
  --out_dir $HOME/dataspace/graph/fully-synthetic/small-world-n1000-${L}/random-d01/dictionaries/ \
  --split 0.2
done

for i in 2, 3, 4, 6, 7
do
  L=k10-p$i

  python utils/split_dict.py --input $HOME/dataspace/graph/fully-synthetic/small-world-n1000-${L}/random-d01/dictionaries/groundtruth \
  --out_dir $HOME/dataspace/graph/fully-synthetic/small-world-n1000-${L}/random-d01/dictionaries/ \
  --split 0.2
done


#semi
for al in ppi/subgraphs/subgraph3
do
for d in 1 2 3 4 5
do 
    home=$HOME

    D=REGAL-d${d}-seed1
    python utils/split_dict.py \
    --input $home/dataspace/graph/${al}/semi-synthetic/${D}/dictionaries/groundtruth \
    --out_dir $home/dataspace/graph/${al}/semi-synthetic/${D}/dictionaries/ \
    --split 0.01
done
done

D=REGAL-d005
python utils/split_dict.py \
--input $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/ \
--split 0.2


D=REGAL-d01
python utils/split_dict.py \
--input $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/ \
--split 0.2


D=REGAL-d05
python utils/split_dict.py \
--input $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/ \
--split 0.2


D=REGAL-d1
python utils/split_dict.py \
--input $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/ \
--split 0.2

home=$HOME
D=REGAL-d2-seed11
python utils/split_dict.py \
--input $home/dataspace/graph/bn-fly-drosophila_medulla_1/${D}/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/bn-fly-drosophila_medulla_1/${D}/dictionaries/ \
--split 0.2


for pfeat in 1 2 3 4 5
do
  for seed in 1
  do
    for d in 05
    do
      D=REGAL-d${d}-pfeats${pfeat}-seed${seed}
        python utils/split_dict.py \
        --input $HOME/dataspace/graph/bn-fly-drosophila_medulla_1/semi-synthetic/${D}/dictionaries/groundtruth \
        --out_dir $HOME/dataspace/graph/bn-fly-drosophila_medulla_1/semi-synthetic/${D}/dictionaries/ \
        --split 0.2
    done
  done

  for seed in 1
  do
    for d in 05
    do
      D=REGAL-d${d}-pfeats${pfeat}-seed${seed}
        python utils/split_dict.py \
        --input $HOME/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/groundtruth \
        --out_dir $HOME/dataspace/graph/econ-mahindas/semi-synthetic/${D}/dictionaries/ \
        --split 0.2
    done
  done


  for seed in 1
  do
    for d in 05
    do
      D=REGAL-d${d}-pfeats${pfeat}-seed${seed}
        python utils/split_dict.py \
        --input $HOME/dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/${D}/dictionaries/groundtruth \
        --out_dir $HOME/dataspace/graph/ppi/subgraphs/subgraph3/semi-synthetic/${D}/dictionaries/ \
        --split 0.2
    done
  done
done
######del_nodes ######

home=$HOME
for i in 1 2 3 4 5
do
    python utils/split_dict.py \
    --input $home/dataspace/graph/econ-mahindas/del-nodes-p$i/dictionaries/groundtruth \
    --out_dir $home/dataspace/graph/econ-mahindas/del-nodes-p$i/dictionaries \
    --split 0.2
done


##### real_dataset ####
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
--input $home/dataspace/graph/econ-mahindas/permut/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/econ-mahindas/permut/dictionaries \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fb-tw-data/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fb-tw-data/dictionaries/ \
--split 0.2
python utils/split_dict.py \
--input $home/dataspace/graph/fq-tw-data/dictionaries/groundtruth \
--out_dir $home/dataspace/graph/fq-tw-data/dictionaries/ \
--split 0.2





home=$HOME
for s in 0.01 0.03 0.1 0.2
do
    for i in 1 2 3 4 5
    do
        python utils/split_dict.py \
        --input $home/dataspace/graph/fq-tw-data/foursquare/del-nodes-p$i-seed1/dictionaries/groundtruth \
        --out_dir $home/dataspace/graph/fq-tw-data/foursquare/del-nodes-p$i-seed1/dictionaries \
        --split ${s}
    done
done


home=$HOME
for s in 0.01 0.03 0.1 0.2
do
    for i in 1 2 3 4 5
    do
        python utils/split_dict.py \
        --input $home/dataspace/graph/pale_facebook/del-nodes-p${i}-seed1/dictionaries/groundtruth \
        --out_dir $home/dataspace/graph/pale_facebook/del-nodes-p${i}-seed1/dictionaries \
        --split ${s}
    done
done



for s in 0.01 0.03 0.1 0.2
do
    for d in 01 05 1 2 3 4 5
    do
        D=REGAL-d${d}-seed1
        python utils/split_dict.py \
        --input $HOME/dataspace/graph/fq-tw-data/foursquare/${D}/dictionaries/groundtruth \
        --out_dir $HOME/dataspace/graph/fq-tw-data/foursqaure/${D}/dictionaries/ \
        --split ${s}
    done
done

D=REGAL-d${d}-seed1
python utils/split_dict.py \
--input $HOME/dataspace/graph/allmovie_tmdb_final/dictionaries/groundtruth \
--out_dir $HOME/dataspace/graph/allmovie_tmdb_final/dictionaries/ \
--split 0.2