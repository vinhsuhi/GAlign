home=$HOME
fully=$home/dataspace/graph/fully-synthetic/

# for i in `ls $fully | grep seed123`
# do
#     python -u network_alignment.py \
#         --source_dataset $fully/$i/graphsage \
#         --target_dataset $fully/$i/random-d01/graphsage \
#         --groundtruth $fully/$i/random-d01/dictionaries/groundtruth \
#         --alignment_matrix_name $i \
#         BigAlign > log/bigalign/$i
# done

# for i in `ls $fully | grep seed234`
# do
#     python -u network_alignment.py \
#         --source_dataset $fully/$i/graphsage \
#         --target_dataset $fully/$i/random-d01/graphsage \
#         --groundtruth $fully/$i/random-d01/dictionaries/groundtruth \
#         --alignment_matrix_name $i \
#         BigAlign > log/bigalign/$i
# done

for i in `ls $fully | grep -P seed1[01].`
do
    python -u network_alignment.py \
        --source_dataset $fully/$i/graphsage \
        --target_dataset $fully/$i/random-d01/graphsage \
        --groundtruth $fully/$i/random-d01/dictionaries/groundtruth \
        --alignment_matrix_name $i \
        BigAlign > log/bigalign/$i
done
