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
#         PALE \
#         --embedding_epochs 100 \
#         --mapping_epochs 100 \
#         --train_dict $fully/small-world-n$n-k10-p5-seed123/random-d01/dictionaries/node,split=0.2.train.dict \
#         --batch_size_embedding 512 \
#         --cuda > output/PALE/small-world-n$n-k10-p5-seed123
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
            PALE \
            --embedding_epochs 1 \
            --mapping_epochs 1 \
            --train_dict $fully/small-world-n$n-k$k-p5-seed123/dictionaries/node,split=0.2.train.dict \
            --batch_size_embedding 512 \
            --cuda > output/PALE/small-world-n$n-k$k-p5-seed123
    done
done
