

for d in 1
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/twitter
    PREFIX2=permut
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    CUDA_VISIBLE_DEVICE=1 python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_twitter \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda 
done

# foursquare-delnodes
ssh -p 19469 vinhtv@0.tcp.ngrok.io



for d in 1
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data
    PREFIX1=
    PREFIX2=tmdb
    TRAINRATIO=0.2
    TRAINP=${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/dictionaries/groundtruth

    CUDA_VISIBLE_DEVICE=1 python -u network_alignment.py \
    --source_dataset ${PD}/${PREFIX1}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name allmovie_tmdb \
    --train_dict ${TRAINP} \
    --test_dict ${TESTP} \
    --mapper linear 
done


for d in 05
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=del-nodes-${X}
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquarex \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --embedding_epochs 1000 \
    --batch_size_embedding 512 \
    --neg_sample_size 5 \
    --cuda 
done



#######facebook del-nodes

for d in 01 05
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/tb-tw-data/facebook
    PREFIX2=del-nodes-${X}
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/facebook_${PREFIX2}_linear
done


# twitter -delnodes

for d in 01 05
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/twitter
    PREFIX2=del-nodes-${X}
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_twitter \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/twitter_${PREFIX2}_linear
done



### ppi-deledges


for d in 01 05
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/ppi
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_ppi \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/ppi_${PREFIX2}_linear
done

### facebook-deledges


for d in 01 05
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/tb-tw-data/facebook
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/facebook_${PREFIX2}_linear
done

### twitter-deledges


for d in 01 05
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/twitter
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_twitter \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/twitter_${PREFIX2}_linear
done

### foursquare-deledges


for d in 01
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/foursquare_${PREFIX2}_linear
done






PD=$HOME/dataspace/graph/ppi
PREFIX2=permut
TRAINRATIO=0.1
TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${TESTP} \
--seed 111 \
NAWAL \
--embedding_name ${PREFIX2}_ppi \
--train_dict ${TRAINP} \
--test_dict ${TEST} \
--load_emb \
--cuda > output/NAWAL/ppi_${PREFIX2}


PD=$HOME/dataspace/graph/fq-tw-data/foursquare
PREFIX2=permut
TRAINRATIO=0.1
TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${TESTP} \
--seed 111 \
NAWAL \
--embedding_name ${PREFIX2}_foursquare \
--train_dict ${TRAINP} \
--test_dict ${TEST} \
--load_emb \
--cuda > output/NAWAL/foursquare_${PREFIX2}



PD=$HOME/dataspace/graph/fb-tw-data/twitter
PREFIX2=permut
TRAINRATIO=0.1
TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${TESTP} \
--seed 111 \
NAWAL \
--embedding_name ${PREFIX2}_twitter \
--train_dict ${TRAINP} \
--test_dict ${TEST} \
--load_emb \
--cuda > output/NAWAL/twitter_${PREFIX2}


PD=$HOME/dataspace/graph/tb-tw-data/facebook
PREFIX2=permut
TRAINRATIO=0.1
TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

python -u network_alignment.py \
--source_dataset ${PD}/graphsage/ \
--target_dataset ${PD}/${PREFIX2}/graphsage/ \
--groundtruth ${TESTP} \
--seed 111 \
NAWAL \
--embedding_name ${PREFIX2}_facebook \
--train_dict ${TRAINP} \
--test_dict ${TEST} \
--load_emb \
--cuda > output/NAWAL/facebook_${PREFIX2}





###############################REMAKE





for d in 1
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/twitter
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_twitter \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/twitter_${PREFIX2}

    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/facebook
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/facebook_${PREFIX2}


    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_twitter \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/foursquare_${PREFIX2}

done




for d in 1
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/twitter
    PREFIX2=del-nodes-${X}
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_twitter \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --suhi "algorithms/NAWAL/embeddings/REGAL-d0-seed1_twitter_source" \
    --load_emb \
    --cuda > output/NAWAL/twitter_${PREFIX2}
done


for d in 0
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/facebook
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --embedding_epochs 1000 \
    --batch_size_embedding 512 \
    --neg_sample_size 5 \
    --cuda > output/NAWAL/facebook_${PREFIX2}
done



for d in 05 1 2 3
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/facebook
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --suhi "algorithms/NAWAL/embeddings/REGAL-d0-seed1_facebook_source" \
    --embedding_epochs 1000 \
    --batch_size_embedding 512 \
    --neg_sample_size 5 \
    --cuda > output/NAWAL/facebook_${PREFIX2}
done



for d in 0 05 1 2 3
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/facebook
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --suhi "algorithms/NAWAL/embeddings/REGAL-d0-seed1_facebook_source" \
    --embedding_epochs 1000 \
    --batch_size_embedding 512 \
    --neg_sample_size 5 \
    --load_emb \
    --cuda > output/NAWAL/facebook_${PREFIX2}
done



for d in 0
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --embedding_epochs 1000 \
    --batch_size_embedding 512 \
    --neg_sample_size 5 \
    --cuda > output/NAWAL/foursquare_${PREFIX2}
done




for d in 0 05 2 4
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --suhi "algorithms/NAWAL/embeddings/REGAL-d0-seed1_foursquare_source" \
    --embedding_epochs 700 \
    --batch_size_embedding 512 \
    --neg_sample_size 15 \
    --cuda > output/NAWAL/foursquare_${PREFIX2}
done


for d in 05 2
do
    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --embedding_epochs 1000 \
    --batch_size_embedding 512 \
    --neg_sample_size 10 \
    --cuda > output/NAWAL/foursquare_${PREFIX2}


    X=d${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-${X}

    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda > output/NAWAL/foursquare_${PREFIX2}
done




for d in 05
do
    # X=p${d}-seed1
    # PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    # PREFIX2=del-nodes-${X}
    # TRAINRATIO=0.1
    # TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    # TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    # TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    # python -u network_alignment.py \
    # --source_dataset ${PD}/graphsage/ \
    # --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    # --groundtruth ${TESTP} \
    # --seed 111 \
    # NAWAL \
    # --embedding_name ${PREFIX2}_foursquare \
    # --train_dict ${TRAINP} \
    # --test_dict ${TEST} \
    # --embedding_epochs 1000 \
    # --batch_size_embedding 512 \
    # --neg_sample_size 10 \
    # --cuda > output/NAWAL/foursquare_${PREFIX2}


    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=del-nodes-${X}
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --load_emb \
    --cuda
done





for d in 4 3 2 1
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-d${d}-seed1
    TRAINRATIO=0.2
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TESTP} \
    --mapper all \
    --pale_map_epochs 100 \
    --embedding_dim 200 \
    --cuda > output/ablation_test
done

    # X=p${d}-seed1
    # PD=$HOME/dataspace/graph/fb-tw-data/twitter
    # PREFIX2=del-nodes-${X}
    # TRAINRATIO=0.1
    # TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    # TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    # TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    # python -u network_alignment.py \
    # --source_dataset ${PD}/graphsage/ \
    # --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    # --groundtruth ${TESTP} \
    # --seed 111 \
    # NAWAL \
    # --embedding_name ${PREFIX2}_twitter \
    # --train_dict ${TRAINP} \
    # --test_dict ${TEST} \
    # --embedding_epochs 1000 \
    # --batch_size_embedding 512 \
    # --neg_sample_size 10 \
    # --cuda 


    # X=p${d}-seed1
    # PD=$HOME/dataspace/graph/fb-tw-data/twitter
    # PREFIX2=REGAL-d${d}-seed1
    # TRAINRATIO=0.1
    # TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    # TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    # TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    # python -u network_alignment.py \
    # --source_dataset ${PD}/graphsage/ \
    # --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    # --groundtruth ${TESTP} \
    # --seed 111 \
    # NAWAL \
    # --embedding_name ${PREFIX2}_twitter \
    # --train_dict ${TRAINP} \
    # --test_dict ${TEST} \
    # --load_emb \
    # --cuda 


    # X=p${d}-seed1
    # PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    # PREFIX2=del-nodes-${X}
    # TRAINRATIO=0.1
    # TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    # TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    # TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    # python -u network_alignment.py \
    # --source_dataset ${PD}/graphsage/ \
    # --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    # --groundtruth ${TESTP} \
    # --seed 111 \
    # NAWAL \
    # --embedding_name ${PREFIX2}_foursquare \
    # --train_dict ${TRAINP} \
    # --test_dict ${TEST} \
    # --embedding_epochs 1000 \
    # --batch_size_embedding 512 \
    # --neg_sample_size 10 \
    # --cuda 


    # X=p${d}-seed1
    # PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    # PREFIX2=REGAL-d${d}-seed1
    # TRAINRATIO=0.1
    # TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    # TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    # TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    # python -u network_alignment.py \
    # --source_dataset ${PD}/graphsage/ \
    # --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    # --groundtruth ${TESTP} \
    # --seed 111 \
    # NAWAL \
    # --embedding_name ${PREFIX2}_foursquare \
    # --train_dict ${TRAINP} \
    # --test_dict ${TEST} \
    # --load_emb \
    # --cuda 
done


python -m tests.semi_synthetic


for model in "fb-tw-data/facebook" "fb-tw-data/twitter" "fq-tw-data/foursquare" 
do
    ppi="/home/bigdata/thomas/dataspace/graph/${model}"
    for i in 0 01 05 1 2 3 4 5
    do
        DIR="$ppi/REGAL-d${i}-seed1"
        python utils/shuffle_graph.py --input_dir $DIR --out_dir $DIR--1
        rm -r $DIR
        mv $DIR--1 $DIR
    done
done



for model in "fb-tw-data/facebook" "fb-tw-data/twitter" "fq-tw-data/foursquare" 
do
    for s in 0.01 0.03 0.1 0.2
    do
        for d in 0 01 05 1 2 3 4
        do
            D=REGAL-d${d}-seed1
            python utils/split_dict.py \
            --input $HOME/dataspace/graph/${model}/${D}/dictionaries/groundtruth \
            --out_dir $HOME/dataspace/graph/${model}/${D}/dictionaries/ \
            --split ${s}
        done
    done
done




for d in 1
do


    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fq-tw-data/foursquare
    PREFIX2=REGAL-d${d}-seed1
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_foursquare \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --pale_map_epochs 100 \
    --embedding_dim 200 \
    --cuda 
done






for d in 1
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/twitter
    PREFIX2=REGAL-d${d}-seed1
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_twitter \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --pale_map_epochs 100 \
    --embedding_dim 200 \
    --cuda 
done




for d in 1
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/fb-tw-data/facebook
    PREFIX2=REGAL-d${d}-seed1
    TRAINRATIO=0.1
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --pale_map_epochs 100 \
    --embedding_dim 200 \
    --cuda 
done



for d in 2
do
    X=p${d}-seed1
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2
    TRAINP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict
    TESTP=${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict
    TEST=${PD}/${PREFIX2}/dictionaries/groundtruth

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${TESTP} \
    --seed 111 \
    NAWAL \
    --embedding_name ${PREFIX2}_facebook \
    --train_dict ${TRAINP} \
    --test_dict ${TEST} \
    --pale_map_epochs 100 \
    --embedding_dim 250 \
    --cuda 
done


