home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

declare -a ks=("4" "10" "100" "200" "350" "500" "700" "900" "999")
declare -a ps=("5")

for k in "${ks[@]}"
do
    for p in "${ps[@]}"
    do
        python -u network_alignment.py \
        --source_dataset $fully/small-world-n1000-k$k-p$p-seed123/graphsage \
        --target_dataset $fully/small-world-n1000-k$k-p$p-seed123/random-d01/graphsage \
        --groundtruth $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/groundtruth \
        REGAL > log/REGAL/smallworld-n1000-k$k-p$p-seed123-01
    done
done
