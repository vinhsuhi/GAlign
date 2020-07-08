home=$HOME
fully="$home/dataspace/graph/fully-synthetic"

# for n in 500 1000 2000 5000 10000 20000 50000
# do
#     python utils/split_dict.py \
#         --input $fully/small-world-n$n-k10-p5-seed123/random-d01/dictionaries/groundtruth \
#         --out_dir $fully/small-world-n$n-k10-p5-seed123/random-d01/dictionaries/ \
#         --split 0.2

#     python -u network_alignment.py \
#         --source_dataset $fully/small-world-n$n-k10-p5-seed123/graphsage \
#         --target_dataset $fully/small-world-n$n-k10-p5-seed123/random-d01/graphsage \
#         --groundtruth $fully/small-world-n$n-k10-p5-seed123/random-d01/dictionaries/node,split=0.2.test.dict \
#         DeepLink \
#         --embedding_epochs 20 \
#         --unsupervised_epochs 100 \
#         --supervised_epochs 500 \
#         --train_dict $fully/small-world-n$n-k10-p5-seed123/random-d01/dictionaries/node,split=0.2.train.dict \
#         --number_walks 10 \
#         --walk_length 5 \
#         --window_size 5 \
#         --top_k 5 \
#         --alpha 0.8 \
#         --fullgt $fully/small-world-n$n-k10-p5-seed123/random-d01/dictionaries/node,split=0.2.train.dict \
#         --num_cores 8 \
#         --cuda > output/DeepLink/small-world-n$n-k10-p5-seed123
# done

for n in 1000 2000 5000 10000 20000
do
    for k in 20 60 100 200 350
    do
        python utils/split_dict.py \
            --input $fully/small-world-n$n-k$k-p5-seed123/dictionaries/groundtruth \
            --out_dir $fully/small-world-n$n-k$k-p5-seed123/dictionaries/ \
            --split 0.2

        python -u network_alignment.py \
            --source_dataset $fully/small-world-n$n-k$k-p5-seed123/graphsage \
            --target_dataset $fully/small-world-n$n-k$k-p5-seed123/graphsage \
            --groundtruth $fully/small-world-n$n-k$k-p5-seed123/dictionaries/node,split=0.2.test.dict \
            DeepLink \
            --embedding_epochs 1 \
            --unsupervised_epochs 1 \
            --supervised_epochs 1 \
            --train_dict $fully/small-world-n$n-k$k-p5-seed123/dictionaries/node,split=0.2.train.dict \
            --number_walks 10 \
            --walk_length 5 \
            --window_size 5 \
            --top_k 5 \
            --alpha 0.8 \
            --fullgt $fully/small-world-n$n-k$k-p5-seed123/dictionaries/node,split=0.2.train.dict \
            --num_cores 8 \
            --cuda > output/DeepLink/small-world-n$n-k$k-p5-seed123
    done
done
