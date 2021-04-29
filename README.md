# GAlign
Code of the paper: ***[Adaptive Network Alignment with Multi-order Convolutional Networks](https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300a085/290300a085.pdf)***.

# Environment

* python>=3.5 
* networkx >= 2.4
* pytorch >= 1.2.0 
* numpy >= 1.18.1 

# Running

For allmv_tmdb dataset

```
python -u network_alignment.py --source_dataset graph_data/allmv_tmdb/allmv/graphsage --target_dataset graph_data/allmv_tmdb/tmdb/graphsage --groundtruth graph_data/allmv_tmdb/dictionaries/groundtruth GAlign --log --GAlign_epochs 10 --refinement_epochs 50 --cuda
```

For douban dataset

```
python -u network_alignment.py --source_dataset graph_data/douban/online/graphsage --target_dataset graph_data/douban/offline/graphsage --groundtruth graph_data/douban/dictionaries/groundtruth GAlign --log --GAlign_epochs 50 --refinement_epochs 0 --cuda --embedding_dim 128 --lr 0.005 --noise_level 0 --refinement_epochs 0 
```

# Citation

Please politely cite our work as follows:

*Huynh Thanh Trung, Tong Van Vinh, Nguyen Thanh Tam, Hongzhi Yin, Matthias Weidlich, Nguyen Quoc Viet Hung. Adaptive Network Alignment with Multi-order Convolutional Networks. In: ICDE 2020*
