home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

declare -a ks=("10" "50" "100")
declare -a ps=("2" "4" "6")

for k in "${ks[@]}"
do
    for p in "${ps[@]}"
    do
        python utils/split_dict.py \
        --input $fully/small-world-n1000-k$k-p$p/random-d1/dictionaries/groundtruth \
        --out_dir $fully/small-world-n1000-k$k-p$p/random-d1/dictionaries/ \
        --split 0.2
        python -u network_alignment.py \
        --source_dataset $fully/small-world-n1000-k$k-p$p/graphsage \
        --target_dataset $fully/small-world-n1000-k$k-p$p/random-d1/graphsage \
        --groundtruth $fully/small-world-n1000-k$k-p$p/random-d1/dictionaries/node,split=0.2.test.dict \
        --alignment_matrix_name smallworld-n1000-k$k-p$p \
        IONE \
        --gt_train $fully/small-world-n1000-k$k-p$p/random-d1/dictionaries/node,split=0.2.train.dict \
        --total_iter 10000000 \
        --dim 200 > log/ione/smallworld-n1000-k$k-p$p
    done
done
