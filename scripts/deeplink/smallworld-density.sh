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
        
        TRAINRATIO=0.2
        NWALKS=10
        WLEN=5
        WSIZE=5
        TOPK=5
        AL=0.8
        EMEP=20
        SEP=500
        USEP=100

        python -u network_alignment.py \
        --source_dataset $fully/small-world-n1000-k$k-p$p-seed123/graphsage \
        --target_dataset $fully/small-world-n1000-k$k-p$p-seed123/random-d01/graphsage \
        --groundtruth $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/node,split=0.2.test.dict \
        DeepLink \
        --embedding_epochs ${EMEP} \
        --unsupervised_epochs ${USEP} \
        --supervised_epochs ${SEP} \
        --train_dict $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/node,split=0.2.train.dict \
        --number_walks ${NWALKS} \
        --walk_length ${WLEN} \
        --window_size ${WSIZE} \
        --top_k ${TOPK} \
        --alpha ${AL} \
        --fullgt $fully/small-world-n1000-k$k-p$p-seed123/random-d01/dictionaries/node,split=0.2.train.dict \
        --num_cores 8 \
        --cuda > log/DeepLink/smallworld-n1000-k$k-p$p-seed123-01
    done
done
