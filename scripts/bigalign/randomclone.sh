home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

for i in `ls $fully | grep seed123`
do
    for r in 02 03 04 05
    do
        python -u network_alignment.py \
            --source_dataset $fully/$i/graphsage \
            --target_dataset $fully/$i/random-d$r/graphsage \
            --groundtruth $fully/$i/random-d$r/dictionaries/groundtruth \
            --alignment_matrix_name $i \
            BigAlign > log/bigalign/$i-$r
    done
done
