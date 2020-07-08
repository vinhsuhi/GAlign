home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

declare -a ks=("10")
declare -a ps=("2" "3" "4" "5" "6" "7")

for k in "${ks[@]}"
do
    for p in "${ps[@]}"
    do
        python -u network_alignment.py \
        --source_dataset $fully/small-world-n1000-k$k-p$p/graphsage \
        --target_dataset $fully/small-world-n1000-k$k-p$p/random-d01/graphsage \
        --groundtruth $fully/small-world-n1000-k$k-p$p/random-d01/dictionaries/groundtruth \
        --alignment_matrix_name smallworld-n1000-k$k-p$p \
        BigAlign > log/bigalign/smallworld-n1000-k$k-p$p
    done
done
