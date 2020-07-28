# GAlign
Code of the paper: ***Adaptive Network Alignment with Multi-order Convolutional Network***.

# Environment

* python>=3.5 
* networkx >= 2.4
* pytorch >= 1.2.0 
* numpy >= 1.18.1 

# Running

```
python -u network_alignment.py --source_dataset graph_data/allmv_tmdb/allmv/graphsage --target_dataset graph_data/allmv_tmdb/tmdb/graphsage --groundtruth graph_data/allmv_tmdb/dictionaries/groundtruth GAlign --log --GAlign_epochs 10 --refinement_epochs 50 --cuda
```

# Citation

Please politely cite our work as follows:

*Huynh Thanh Trung, Tong Van Vinh, Nguyen Thanh Tam, Hongzhi Yin, Matthias Weidlich, Nguyen Quoc Viet Hung. Adaptive Network Alignment with Multi-order Convolutional Network. In: ICDE 2020*
