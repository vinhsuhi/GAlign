# GAlign
Code of the paper: ***Entity Alignment for Knowledge Graphs with Multi-order Convolutional Network***.

# Environment

* python>=3.5 
* networkx >= 2.4
* pytorch >= 1.2.0 
* numpy >= 1.18.1 

# Running

```
python -u network_alignment.py --dataset_name zh_en --source_dataset data/networkx/zh_enID/zh/graphsage/ --target_dataset data/networkx/zh_enID/en/graphsage --groundtruth data/networkx/zh_enID/dictionaries/groundtruth EMGCN --sparse --log 
```

# Citation

Please politely cite our work as follows:

*Huynh Thanh Trung, Tong Van Vinh, Nguyen Thanh Tam, Hongzhi Yin, Matthias Weidlich, Nguyen Quoc Viet Hung. Adaptive Network Alignment with Multi-order Convolutional Networks. In: ICDE 2020*
