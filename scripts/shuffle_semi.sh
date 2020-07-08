ppi="/home/bigdata/thomas/dataspace/graph/ppi/subgraphs/subgraph3"

for s in {1..20}
do
    for i in 1 2 3 4 5
    do
        DIR="$ppi/del-nodes-p${i}-seed${s}"
        python utils/shuffle_graph.py --input_dir ${DIR} --out_dir $DIR--1
        rm -r $DIR
        mv $DIR--1 $DIR
    done
done

DIR=/home/bigdata/thomas/dataspace/graph/econ-mahindas
python utils/shuffle_graph.py --input_dir ${DIR} --out_dir ${DIR}/permut