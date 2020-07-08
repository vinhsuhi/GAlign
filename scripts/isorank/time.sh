home=$HOME
fully="$home/dataspace/graph/fully-synthetic"

# for n in 500 1000 2000 5000 10000 20000 50000
# do
#     python -u network_alignment.py \
#         --source_dataset $fully/small-world-n$n-k10-p5-seed123/graphsage \
#         --target_dataset $fully/small-world-n$n-k10-p5-seed123/random-d01/graphsage \
#         --groundtruth $fully/small-world-n$n-k10-p5-seed123/random-d01/dictionaries/groundtruth \
#         IsoRank > log/IsoRank/small-world-n$n-k10-p5-seed123
# done

for n in 1000 2000 5000 10000 20000
do
    for k in 20 60 100 200 350
    do
        python -u network_alignment.py \
            --source_dataset $fully/small-world-n$n-k$k-p5-seed123/graphsage \
            --target_dataset $fully/small-world-n$n-k$k-p5-seed123/graphsage \
            --groundtruth $fully/small-world-n$n-k$k-p5-seed123/dictionaries/groundtruth \
            IsoRank > log/IsoRank/small-world-n$n-k$k-p5-seed123
    done
done