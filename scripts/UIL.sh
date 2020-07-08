
for lr in 0.01
do
    for msab in 2
    do
        for ee in 1
        do 
            for ls in g l gl
            do
                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
                TRAINRATIO=0.2

                python -u network_alignment.py \
                --source_dataset ${PD}/graphsage/ \
                --target_dataset ${PD}/${PREFIX2}/graphsage/ \
                --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
                GAlign \
                --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
                --lr ${lr} \
                --emb_epochs ${ee} \
                --jump \
                --separate_emb \
                --loss ${ls} \
                --neg_sample_size 10 \
                --num_MSA_blocks ${msab} \
                --cuda

                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
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
                --cuda > output/GAlign/econ_d2_ee${ee}_ls${ls}_msab${msab}_jump

            done
        done
    done
done

for lr in 0.01
do
    for msab in 2 3 4
    do
        for ee in 1 2 5 10 20
        do 
            for ls in g l gl
            do
                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
                TRAINRATIO=0.2

                python -u network_alignment.py \
                --source_dataset ${PD}/graphsage/ \
                --target_dataset ${PD}/${PREFIX2}/graphsage/ \
                --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
                GAlign \
                --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
                --lr ${lr} \
                --emb_epochs ${ee} \
                --loss ${ls} \
                --neg_sample_size 10 \
                --num_MSA_blocks ${msab} \
                --cuda

                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
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
                --cuda > output/GAlign/econ_d2_ee${ee}_ls${ls}_msab${msab}

            done
        done
    done
done


for lr in 0.01
do
    for msab in 1 2 3 4
    do
        for ee in 1 2 5 10 20
        do 
            for ls in g l gl
            do
                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
                TRAINRATIO=0.2

                python -u network_alignment.py \
                --source_dataset ${PD}/graphsage/ \
                --target_dataset ${PD}/${PREFIX2}/graphsage/ \
                --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
                GAlign \
                --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
                --lr ${lr} \
                --emb_epochs ${ee} \
                --loss ${ls} \
                --separate_emb \
                --neg_sample_size 10 \
                --num_MSA_blocks ${msab} \
                --cuda

                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
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
                --cuda > output/GAlign/econ_d2_ee${ee}_ls${ls}_msab${msab}_sep

            done
        done
    done
done




for lr in 0.01
do
    for msab in 1 2 3 4
    do
        for ee in 1 2 5 10 20
        do 
            for ls in g l gl
            do
                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
                TRAINRATIO=0.2

                python -u network_alignment.py \
                --source_dataset ${PD}/graphsage/ \
                --target_dataset ${PD}/${PREFIX2}/graphsage/ \
                --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
                GAlign \
                --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
                --lr ${lr} \
                --emb_epochs ${ee} \
                --loss ${ls} \
                --separate_emb \
                --jump \
                --neg_sample_size 10 \
                --num_MSA_blocks ${msab} \
                --cuda

                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
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
                --cuda > output/GAlign/econ_d2_ee${ee}_ls${ls}_msab${msab}_sep_jump

            done
        done
    done
done








PD=$HOME/dataspace/graph/econ-mahindas
PREFIX2=semi-synthetic/REGAL-d2-seed1
TRAINRATIO=0.2

python -u network_alignment.py \
--source_dataset graphsage/ \
--target_dataset graphsage/ \
--groundtruth a \
--syn \
GAlign \
--lr 0.01 \
--neg_sample_size 10 \
--num_MSA_blocks 2 \
--emb_epochs 10 \
--cuda




parser_GAlign = subparsers.add_parser("GAlign", help="GAlign algorithm")
    parser_GAlign.add_argument('--cuda',                action="store_true")
    
    parser_GAlign.add_argument('--embedding_dim',       default=300,         type=int)
    parser_GAlign.add_argument('--emb_epochs',    default=500,        type=int)
    parser_GAlign.add_argument('--lr', default=0.1, type=float)
    parser_GAlign.add_argument('--neg_sample_size',  default=5, type=int)
    parser_GAlign.add_argument('--loss', type=str, default='n')
    parser_GAlign.add_argument('--num_MSA_blocks', type=int, default=2)
    parser_GAlign.add_argument('--mapping_weight', type=float, default=10)
    parser_GAlign.add_argument('--model_type', type=int, default=1)
    parser_GAlign.add_argument('--input_type', type=str, default='mt') # mt t r
    parser_GAlign.add_argument('--act', type=str, default='tanh')
    parser_GAlign.add_argument('--train_dict',          default="/home/bigdata/thomas/dataspace/graph/econ-mahindas/semi-synthetic/dictionaries/node,split=0.2.train.dict")
    parser_GAlign.add_argument('--full_dict', default="/home/bigdata/thomas/dataspace/graph/econ-mahindas/semi-synthetic/dictionaries/groundtruth")
    parser_GAlign.add_argument('--log', action="store_true")
    parser_GAlign.add_argument('--invest', action="store_true")
    parser_GAlign.add_argument('--input_dim', default=100)
    parser_GAlign.add_argument('--loss_type', default=1, type=int, help="1 if normal, 2 if negative")
    parser_GAlign.add_argument('--linkpred', action='store_true')
    



for lr in 0.01
do
    for msab in 2 3 4
    do
            for ls in n
            do
                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
                TRAINRATIO=0.2

                python -u network_alignment.py \
                --source_dataset ${PD}/graphsage/ \
                --target_dataset ${PD}/${PREFIX2}/graphsage/ \
                --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
                GAlign \
                --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
                --lr ${lr} \
                --jump \
                --loss ${ls} \
                --separate_emb \
                --neg_sample_size 10 \
                --num_MSA_blocks ${msab} \
                --cuda

                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
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
                --cuda > output/GAlign/econ_d2_ls${ls}_msab${msab}_sep_jump

            done
        done
    done
done



for lr in 0.01
do
    for msab in 2 3 4
    do
            for ls in m
            do
                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
                TRAINRATIO=0.2

                python -u network_alignment.py \
                --source_dataset ${PD}/graphsage/ \
                --target_dataset ${PD}/${PREFIX2}/graphsage/ \
                --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
                GAlign \
                --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
                --lr ${lr} \
                --loss ${ls} \
                --separate_emb \
                --neg_sample_size 10 \
                --num_MSA_blocks ${msab} \
                --cuda

                PD=$HOME/dataspace/graph/econ-mahindas
                PREFIX2=semi-synthetic/REGAL-d2-seed1
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
                --cuda > output/GAlign/econ_d2_ls${ls}_msab${msab}_sep

            done
        done
    done
done






0.8521

Accuracy: 0.8551
MAP: 0.7524
Precision_5: 0.8801
Precision_10: 0.9241

Use linear at last layer
Accuracy: 0.8721
MAP: 0.7366
Precision_5: 0.8871
Precision_10: 0.9281


No Jump, No nn
Accuracy: 0.8811
MAP: 0.7776
Precision_5: 0.8841
Precision_10: 0.9221

* Random input embedding 
* Train and mapped input embedding 
    * Stack without weight and activate is best
    * ReLU and Tanh
    * 
* Train but not yet mapped input embedding 


3 layer : 73 %
3 layer without first: 78.22%
4 layer without first: 77.82%

2 layer : 68 %
2 layer without first: 74%

 0 is: 0.8871, Final acc is: 0.8871
0.00001
Acc layer 0 is: 0.8232, Acc layer 1 is: 0.9291, Acc layer 2 is: 0.8841, Final acc is: 0.9391

Acc layer 0 is: 0.7692, Acc layer 1 is: 0.8492, Acc layer 2 is: 0.7912, Final acc is: 0.8751
Full_time:  8.918692588806152
Acc layer 0 is: 0.1638, Acc layer 1 is: 0.7692, Acc layer 2 is: 0.8791, Acc layer 3 is: 0.7912, Final acc is: 0.9211
Acc layer 0 is: 0.1638, Acc layer 1 is: 0.8012, Acc layer 2 is: 0.8871, Acc layer 3 is: 0.8392, Final acc is: 0.9231
Acc layer 0 is: 0.1528, Acc layer 1 is: 0.6784, Acc layer 2 is: 0.6854, Acc layer 3 is: 0.4784, Final acc is: 0.8302
Acc layer 0 is: 0.1528, Acc layer 1 is: 0.6814, Acc layer 2 is: 0.7156, Acc layer 3 is: 0.5829, Final acc is: 0.8422
Acc layer 0 is: 0.1528, Acc layer 1 is: 0.6784, Acc layer 2 is: 0.7246, Acc layer 3 is: 0.6231, Final acc is: 0.8563




Acc layer 0 is: 0.1528, Acc layer 1 is: 0.7126, Acc layer 2 is: 0.7528, Acc layer 3 is: 0.6503, Final acc is: 0.8623
Acc layer 0 is: 0.1528, Acc layer 1 is: 0.6814, Acc layer 2 is: 0.7397, Acc layer 3 is: 0.6121, Final acc is: 0.8533

RELU
Acc layer 0 is: 0.1625, Acc layer 1 is: 0.4958, Acc layer 2 is: 0.4748, Acc layer 3 is: 0.3669, Final acc is: 0.7086
TANH
Acc layer 0 is: 0.1625, Acc layer 1 is: 0.5241, Acc layer 2 is: 0.5314, Acc layer 3 is: 0.4308, Final acc is: 0.7327



RELU
Acc layer 0 is: 0.1625, Acc layer 1 is: 0.5010, Acc layer 2 is: 0.4853, Acc layer 3 is: 0.3669, Final acc is: 0.7138
TANH
Acc layer 0 is: 0.1625, Acc layer 1 is: 0.5294, Acc layer 2 is: 0.5398, Acc layer 3 is: 0.4245, Final acc is: 0.7317

Douban 30%

# 75 31.96
# REGAL_D2_20feats
# 500 epochs, tanh, loss 1
# Layer: 0, edge source: 1.9600, edge target: 1.9600, non edge source: 1.9400, non edge target: 1.9400
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 1.2593, edge target: 1.2703, non edge source: 1.5883, non edge target: 1.6374
# Layer: 1, anchor distance1: 0.1141, anchor distance2: 0.1442
# Layer: 2, edge source: 1.7055, edge target: 1.6064, non edge source: 2.0058, non edge target: 2.0103
# Layer: 2, anchor distance1: 0.1108, anchor distance2: 0.1243
# Layer: 3, edge source: 1.4270, edge target: 1.3209, non edge source: 2.0016, non edge target: 2.0089
# Layer: 3, anchor distance1: 0.1235, anchor distance2: 0.1520
# Acc layer 0 is: 0.0170, Acc layer 1 is: 0.8042, Acc layer 2 is: 0.8362, Acc layer 3 is: 0.7493, Final acc is: 0.9301
# Layer: 1 MEAN source: 26.05, target: 21.4. STD source: 20.80498738283684, target: 17.367786272291585
# Layer: 2 MEAN source: 12.639408866995074, target: 10.466009852216748. STD source: 18.214324023980733, target: 15.003795163237717
# Layer: 3 MEAN source: 12.350190839694656, target: 10.145038167938932. STD source: 18.08953769473896, target: 14.823386277222017
# Layer: 4 MEAN source: 12.213756613756614, target: 10.095238095238095. STD source: 17.69198010291463, target: 14.44659673481761
# Full_time:  12.757715463638306

# REGAL_D2_20feats
# 500 epochs, tanh, loss 1, linear
# Layer: 0, edge source: 1.9600, edge target: 1.9600, non edge source: 1.9400, non edge target: 1.9400
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 1.8057, edge target: 1.7943, non edge source: 2.0212, non edge target: 1.9916
# Layer: 1, anchor distance1: 0.1560, anchor distance2: 0.2094
# Layer: 2, edge source: 1.6678, edge target: 1.5852, non edge source: 2.0190, non edge target: 2.0057
# Layer: 2, anchor distance1: 0.1049, anchor distance2: 0.1237
# Layer: 3, edge source: 1.3867, edge target: 1.3017, non edge source: 2.0286, non edge target: 2.0295
# Layer: 3, anchor distance1: 0.1297, anchor distance2: 0.1522
# Acc layer 0 is: 0.0170, Acc layer 1 is: 0.7842, Acc layer 2 is: 0.8222, Acc layer 3 is: 0.7183, Final acc is: 0.9201
# Layer: 1 MEAN source: 26.05, target: 21.4. STD source: 20.80498738283684, target: 17.367786272291585
# Layer: 2 MEAN source: 12.566801619433198, target: 10.44838056680162. STD source: 18.397120415106617, target: 15.16042309935726
# Layer: 3 MEAN source: 12.176470588235293, target: 10.009643201542913. STD source: 17.723329233388814, target: 14.542456168266895
# Layer: 4 MEAN source: 12.292629262926292, target: 10.179317931793179. STD source: 17.82854908269058, target: 14.513897258014165



# REGAL_D2_20feats
# 500 epochs, relu, loss 1
# Layer: 0, edge source: 1.9600, edge target: 1.9600, non edge source: 1.9400, non edge target: 1.9400
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 0.9253, edge target: 0.9231, non edge source: 1.1108, non edge target: 1.1243
# Layer: 1, anchor distance1: 0.1021, anchor distance2: 0.1251
# Layer: 2, edge source: 1.5336, edge target: 1.4167, non edge source: 1.7418, non edge target: 1.6754
# Layer: 2, anchor distance1: 0.4432, anchor distance2: 0.3971
# Layer: 3, edge source: 1.6109, edge target: 1.5626, non edge source: 1.8214, non edge target: 1.8229
# Layer: 3, anchor distance1: 0.5054, anchor distance2: 0.6414
# Acc layer 0 is: 0.0170, Acc layer 1 is: 0.7882, Acc layer 2 is: 0.5924, Acc layer 3 is: 0.4426, Final acc is: 0.8661
# Layer: 1 MEAN source: 26.05, target: 21.4. STD source: 20.80498738283684, target: 17.367786272291585
# Layer: 2 MEAN source: 12.51048951048951, target: 10.378621378621379. STD source: 18.025183289706444, target: 14.85532973972058
# Layer: 3 MEAN source: 8.416116248348745, target: 7.058124174372523. STD source: 11.860145141457869, target: 9.695457987955265
# Layer: 4 MEAN source: 9.390681003584229, target: 7.899641577060932. STD source: 13.724297028515346, target: 11.009313418664492
# Full_time:  13.548038244247437




# REGAL_D5_20feats
# 500 ee, tanh, loss 1
# Layer: 0, edge source: 1.8600, edge target: 1.8600, non edge source: 1.9200, non edge target: 1.9200
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 1.1894, edge target: 1.1649, non edge source: 1.6301, non edge target: 1.7500
# Layer: 1, anchor distance1: 0.3973, anchor distance2: 0.4212
# Layer: 2, edge source: 1.4448, edge target: 1.2618, non edge source: 1.9822, non edge target: 1.8892
# Layer: 2, anchor distance1: 0.3604, anchor distance2: 0.3795
# Layer: 3, edge source: 1.2905, edge target: 1.0607, non edge source: 1.9707, non edge target: 2.0054
# Layer: 3, anchor distance1: 0.4035, anchor distance2: 0.4091
# Acc layer 0 is: 0.0189, Acc layer 1 is: 0.4057, Acc layer 2 is: 0.4130, Acc layer 3 is: 0.2830, Final acc is: 0.6646
# Layer: 1 MEAN source: 41.15, target: 21.2. STD source: 31.730545220654495, target: 16.842208881260202
# Layer: 2 MEAN source: 11.52965235173824, target: 6.67280163599182. STD source: 17.688500998340764, target: 9.408220773462213
# Layer: 3 MEAN source: 8.303212851405622, target: 4.732931726907631. STD source: 10.146043490589188, target: 5.315635782712662
# Layer: 4 MEAN source: 10.0, target: 5.759312320916905. STD source: 14.485245927156383, target: 7.6510167938529285
# Full_time:  12.804054737091064

# Linear is worse 1% than non

# Douban
# tanh, 500, linear, loss 1
# Layer: 0, edge source: 1.0600, edge target: 1.0600, non edge source: 1.6800, non edge target: 1.6800
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 0.7457, edge target: 0.5242, non edge source: 1.8289, non edge target: 1.8025
# Layer: 1, anchor distance1: 0.2618, anchor distance2: 0.2844
# Layer: 2, edge source: 0.4963, edge target: 0.3225, non edge source: 1.7713, non edge target: 1.7907
# Layer: 2, anchor distance1: 0.1844, anchor distance2: 0.1906
# Layer: 3, edge source: 0.4410, edge target: 0.1991, non edge source: 1.7278, non edge target: 1.7124
# Layer: 3, anchor distance1: 0.2189, anchor distance2: 0.2495
# Acc layer 0 is: 0.0782, Acc layer 1 is: 0.2000, Acc layer 2 is: 0.1687, Acc layer 3 is: 0.1453, Final acc is: 0.3017 (no linear 0.3151)
# Layer: 1 MEAN source: 7.7926829268292686, target: 2.682926829268293. STD source: 12.207292707268389, target: 3.2754744537372087
# Layer: 2 MEAN source: 6.244541484716157, target: 4.021834061135372. STD source: 6.536530093491972, target: 4.34981846215299
# Layer: 3 MEAN source: 5.301980198019802, target: 3.242574257425743. STD source: 5.838833468839097, target: 3.7317966510095153
# Layer: 4 MEAN source: 5.2176470588235295, target: 3.2705882352941176. STD source: 5.841508195070823, target: 3.7601258905524384
# Full_time:  24.510856866836548







for lr in 0.0001
do
    for msab in 3
    do
        for ee in 500
        do
            ED=100
            PD=$HOME/dataspace/graph/econ-mahindas
            PREFIX2=semi-synthetic/REGAL-d5-seed1
            TRAINRATIO=0.2

            python -u network_alignment.py \
            --source_dataset ${PD}/graphsage \
            --target_dataset ${PD}/${PREFIX2}/graphsage/ \
            --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
            GAlign \
            --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
            --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
            --lr ${lr} \
            --neg_sample_size 5 \
            --embedding_dim ${ED} \
            --emb_epochs ${ee} \
            --model_type 6 \
            --act tanh \
            --loss_type 2 \
            --embedding_dim 100 \
            --num_MSA_blocks ${msab} \
            --input_type 'r' \
            --invest \
            --log \
            --cuda 
        done
    done
done







for lr in 0.00005
do
    for msab in 3
    do
        for ee in 500
        do
            ED=100
            PD=$HOME/dataspace/graph/douban
            PREFIX1=online
            PREFIX2=offline
            TRAINRATIO=0.2

            python -u network_alignment.py \
            --source_dataset ${PD}/${PREFIX1}/graphsage/ \
            --target_dataset ${PD}/${PREFIX2}/graphsage/ \
            --groundtruth ${PD}/dictionaries/groundtruth \
            FINAL \
            --max_iter 30  \
            --alpha 0.6 \
            --H ${PD}/H.mat 
        done
    done
done


# FINAL
43.8
GCN: 44.63
# 50k epochs DOUBAN
# 43.13 (3 hours)
# Layer: 0, edge source: 1.1200, edge target: 1.1200, non edge source: 1.8000, non edge target: 1.8000
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 0.9285, edge target: 0.5987, non edge source: 1.9751, non edge target: 1.9463
# Layer: 1, anchor distance1: 0.3428, anchor distance2: 0.3683
# Layer: 2, edge source: 0.6839, edge target: 0.3928, non edge source: 1.9854, non edge target: 1.9416
# Layer: 2, anchor distance1: 0.4495, anchor distance2: 0.4350
# Layer: 3, edge source: 0.6300, edge target: 0.3020, non edge source: 1.9955, non edge target: 1.9370
# Layer: 3, anchor distance1: 0.5368, anchor distance2: 0.5295
# Acc layer 0 is: 0.0782, Acc layer 1 is: 0.2525, Acc layer 2 is: 0.2715, Acc layer 3 is: 0.2190, Final acc is: 0.4313
# Layer: 1 MEAN source: 7.7926829268292686, target: 2.682926829268293. STD source: 12.207292707268389, target: 3.2754744537372087
# Layer: 2 MEAN source: 7.629370629370629, target: 4.255244755244755. STD source: 8.20726125855211, target: 4.490225255172927
# Layer: 3 MEAN source: 6.446601941747573, target: 3.5080906148867315. STD source: 6.734533745099189, target: 3.768043870057204
# Layer: 4 MEAN source: 6.412698412698413, target: 3.5238095238095237. STD source: 7.333625363524999, target: 4.130249013812578
# Full_time:  11124.170053958893



# Douban
# Layer: 0, edge source: 1.0600, edge target: 1.0600, non edge source: 1.6800, non edge target: 1.6800
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 0.8366, edge target: 0.6058, non edge source: 1.8595, non edge target: 1.8326
# Layer: 1, anchor distance1: 0.3191, anchor distance2: 0.3638
# Layer: 2, edge source: 0.6010, edge target: 0.3970, non edge source: 1.7425, non edge target: 1.7834
# Layer: 2, anchor distance1: 0.2811, anchor distance2: 0.3047
# Layer: 3, edge source: 0.4916, edge target: 0.2689, non edge source: 1.7873, non edge target: 1.7470
# Layer: 3, anchor distance1: 0.3040, anchor distance2: 0.3799
# Acc layer 0 is: 0.0782, Acc layer 1 is: 0.2101, Acc layer 2 is: 0.2000, Acc layer 3 is: 0.1587, Final acc is: 0.3240
# Layer: 1 MEAN source: 7.7926829268292686, target: 2.682926829268293. STD source: 12.207292707268389, target: 3.2754744537372087
# Layer: 2 MEAN source: 6.421276595744681, target: 4.080851063829787. STD source: 6.837099552090559, target: 4.232329387187132
# Layer: 3 MEAN source: 6.948936170212766, target: 3.753191489361702. STD source: 10.47009667578008, target: 4.200144428308504
# Layer: 4 MEAN source: 5.489130434782608, target: 3.489130434782609. STD source: 6.2214470657301595, target: 3.8320574710261894

# Econ-mahindas
# Layer: 0, edge source: 1.0878, edge target: 1.0878, non edge source: 1.0667, non edge target: 1.0667
# Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
# Layer: 1, edge source: 1.6639, edge target: 1.6513, non edge source: 1.9874, non edge target: 1.8788
# Layer: 1, anchor distance1: 0.2334, anchor distance2: 0.1577
# Layer: 2, edge source: 1.5511, edge target: 1.5071, non edge source: 2.0005, non edge target: 2.0036
# Layer: 2, anchor distance1: 0.1433, anchor distance2: 0.1219
# Layer: 3, edge source: 1.4506, edge target: 1.3532, non edge source: 2.0233, non edge target: 2.0218
# Layer: 3, anchor distance1: 0.1568, anchor distance2: 0.1216
# Acc layer 0 is: 0.2128, Acc layer 1 is: 0.7652, Acc layer 2 is: 0.8422, Acc layer 3 is: 0.7493, Final acc is: 0.9191
# Layer: 1 MEAN source: 15.252747252747254, target: 12.35897435897436. STD source: 22.834614849165842, target: 18.83967644286573
# Layer: 2 MEAN source: 11.914556962025317, target: 9.875527426160337. STD source: 16.913782333369287, target: 14.070022005547452
# Layer: 3 MEAN source: 12.014312977099237, target: 9.831106870229007. STD source: 17.27618008102838, target: 14.172166097834491
# Layer: 4 MEAN source: 11.947537473233405, target: 9.881156316916488. STD source: 17.29837750731671, target: 14.19540174784939





for lr in 0.0001
do
    for msab in 3
    do
        for ee in 300
        do
            PD=$HOME/dataspace/graph/fb-tw-data
            PREFIX1=facebook
            PREFIX2=twitter
            TRAINRATIO=0.2

            python -u network_alignment.py \
            --source_dataset ${PD}/${PREFIX1}/graphsage/ \
            --target_dataset ${PD}/${PREFIX2}/graphsage/ \
            --groundtruth ${PD}/dictionaries/node,split=${TRAINRATIO}.test.dict \
            GAlign \
            --train_dict ${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict \
            --full_dict ${PD}/dictionaries/groundtruth \
            --lr ${lr} \
            --neg_sample_size 5 \
            --emb_epochs ${ee} \
            --model_type 6 \
            --act tanh \
            --loss_type 2 \
            --num_MSA_blocks ${msab} \
            --embedding_dim 200 \
            --log \
            --cuda 
        done
    done
done

PD="/home/bigdata/thomas/dataspace/graph/bn"
python -m utils.fix_features \
--path ${PD} \
--num_feat 20







for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/bn
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.00005 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 100 \
    --num_MSA_blocks 3 \
    --log \
    --cuda > output/GAlign/bn_d${d}
done




for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.00005 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 100 \
    --num_MSA_blocks 3 \
    --log \
    --cuda > output/GAlign/econ_d${d}
done



for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.00005 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 100 \
    --num_MSA_blocks 3 \
    --log \
    --cuda > output/GAlign/ppi_d${d}
done





for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/bn
    PREFIX2=semi-synthetic/del-nodes-p${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.00005 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 100 \
    --num_MSA_blocks 3 \
    --log \
    --cuda > output/GAlign/bn_d${d}_del_nodes
done



for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/bn
    PREFIX2=semi-synthetic/del-nodes-p${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.01 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 90 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest > output/GAlign/bn_del_nodes_d${d}
done



for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/bn
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.01 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 90 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest > output/GAlign/bn_del_edges_d${d}
done



for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/del-nodes-p${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.01 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 90 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest > output/GAlign/econ_del_nodes_d${d}
done



for d in 5
do
    ED=100
    PD=$HOME/dataspace/graph/econ-mahindas
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.01 \
    --neg_sample_size 5 \
    --emb_epochs 1000 \
    --act tanh \
    --embedding_dim 90 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest
done


for d in 1 2 3 4 5
do
    ED=100
    PD=$HOME/dataspace/graph/ppi/subgraphs/subgraph3
    PREFIX2=semi-synthetic/REGAL-d${d}-seed1
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/graphsage \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/${PREFIX2}/dictionaries/groundtruth \
    --lr 0.01 \
    --neg_sample_size 5 \
    --emb_epochs 500 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 90 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest > output/GAlign/ppi_del_edges_d${d}
done



for d in 1 
do
    ED=100
    PD=$HOME/dataspace/graph/douban
    PREFIX1=online
    PREFIX2=offline
    TRAINRATIO=0.2

    python -u network_alignment.py \
    --source_dataset ${PD}/${PREFIX1}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/dictionaries/groundtruth \
    GAlign \
    --train_dict ${PD}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --full_dict ${PD}/dictionaries/groundtruth \
    --lr 0.01 \
    --neg_sample_size 5 \
    --emb_epochs 100 \
    --act tanh \
    --loss_type 2 \
    --embedding_dim 75 \
    --num_MSA_blocks 3 \
    --log \
    --cuda \
    --invest 
done




python -m graphsage.supervised_train --prefix dataspace/graph/music --epochs 1 --max_degree 25 --model graphsage_mean 