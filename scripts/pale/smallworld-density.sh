home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

declare -a ks=("4" "10" "100" "200" "350" "500" "700" "900" "999")
declare -a ps=("5")

for k in "${ks[@]}"
do
    for p in "${ps[@]}"
    do
        python utils/split_dict.py \
        --input $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/groundtruth \
        --out_dir $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/ \
        --split 0.2

        python -u network_alignment.py \
        --source_dataset $fully/small-world-n1000-k$k-p$p-seed123/graphsage \
        --target_dataset $fully/small-world-n1000-k$k-p$p-seed123/random-d01/graphsage \
        --groundtruth $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/node,split=0.2.test.dict \
        PALE \
        --embedding_epochs 100 \
        --mapping_epochs 100 \
        --train_dict $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/node,split=0.2.train.dict \
        --batch_size_embedding 512 \
        --cuda > log/PALE/smallworld-n1000-k$k-p$p-seed123-01
    done
done