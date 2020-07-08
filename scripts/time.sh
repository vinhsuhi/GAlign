python -m tests.fully_synthetic



198917.90
198917.90
80594.16
80594.16
547618.72

for n in 35000 
do
for m in 10
do
for algorithm in "PALE" "DeepLink" "REGAL" "FINAL" "IsoRank" "BigAlign" "IONE" 
do
    python -m suhi \
    --source_dataset dataspace/n${n}-m${m}/graphsage \
    --target_dataset dataspace/n${n}-m${m}/graphsage \
    --train_dict dataspace/n${n}-m${m}/train_dict \
    --test_dict dataspace/n${n}-m${m}/groundtruth \
    --algorithm ${algorithm} \
    --name n${n}_m${m}  > output/${algorithm}/suhi_m${m}_n${n}

    rm -rf temp/*
done
done
done




for n in 34000 35000 33000
do
for m in 5 10 20 30 50
do
for algorithm in "IONE"
do
    python -m suhi \
    --source_dataset dataspace/n${n}-m${m}/graphsage \
    --target_dataset dataspace/n${n}-m${m}/graphsage \
    --train_dict dataspace/n${n}-m${m}/train_dict \
    --test_dict dataspace/n${n}-m${m}/groundtruth \
    --algorithm ${algorithm} \
    --name n${n}_m${m} > output/${algorithm}/suhi_m${m}_n${n}

    rm -rf temp/*
done
done
done



for n in 1000000
do
for m in 5 10 20 30 50
do
for algorithm in "PALE" "DeepLink"
do
    python -m suhi \
    --source_dataset dataspace/n${n}-m${m}/graphsage \
    --target_dataset dataspace/n${n}-m${m}/graphsage \
    --train_dict dataspace/n${n}-m${m}/train_dict \
    --test_dict dataspace/n${n}-m${m}/groundtruth \
    --algorithm ${algorithm} \
    --name n${n}_m${m} > output/${algorithm}/suhi_m${m}_n${n}_Smat

    rm -rf temp/*
done
done
done




for n in 100000 1000000
do
for m in 5 10 20 30 50
do
for algorithm in "REGAL"
do
    python -m suhi \
    --source_dataset dataspace/n${n}-m${m}/graphsage \
    --target_dataset dataspace/n${n}-m${m}/graphsage \
    --train_dict dataspace/n${n}-m${m}/train_dict \
    --test_dict dataspace/n${n}-m${m}/groundtruth \
    --algorithm ${algorithm} \
    --name n${n}_m${m} > output/${algorithm}/suhi_m${m}_n${n}_Smat

    rm -rf temp/*
done
done
done




for n in 35000
do
for m in 5 10 20 30 50
do
for algorithm in "PALE" "DeepLink"
do
    python -m suhi \
    --source_dataset dataspace/n${n}-m${m}/graphsage \
    --target_dataset dataspace/n${n}-m${m}/graphsage \
    --train_dict dataspace/n${n}-m${m}/train_dict \
    --test_dict dataspace/n${n}-m${m}/groundtruth \
    --algorithm ${algorithm} \
    --name n${n}_m${m} > output/${algorithm}/suhi_m${m}_n${n}

    rm -rf temp/*
done
done
done

