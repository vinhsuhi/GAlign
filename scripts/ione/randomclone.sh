home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

for i in `ls $fully | grep seed123`
do
    for r in 02 03 04 05
    do
       python utils/split_dict.py \
            --input $fully/$i/random-d$r/dictionaries/groundtruth \
            --out_dir $fully/$i/random-d$r/dictionaries/ \
            --split 0.2
            python -u network_alignment.py \
            --source_dataset $fully/$i/graphsage \
            --target_dataset $fully/$i/random-d$r/graphsage \
            --groundtruth $fully/$i/random-d$r/dictionaries/node,split=0.2.test.dict \
            --alignment_matrix_name $i \
            IONE \
            --gt_train $fully/$i/random-d01/dictionaries/node,split=0.2.train.dict \
            --total_iter 10000000 \
            --dim 200 > log/ione/$i-$r
    done
done
