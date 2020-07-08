from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.DeepLink.mapping_model import MappingModel
from algorithms.DeepLink.embedding_model import DeepWalk

from input.dataset import Dataset
from utils.graph_utils import load_gt

import numpy as np
import torch.nn as nn
import torch
import networkx as nx

import argparse
import time


class DeepLink(NetworkAlignmentModel):
    def __init__(self, source_dataset, target_dataset, args):
        """
        Parameters
        ----------
        source_dataset: Dataset
            Dataset object of source dataset
        target_dataset: Dataset
            Dataset object of target dataset
        args: argparse.ArgumentParser.parse_args()
            arguments as parameters for model.
        """
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        super(DeepLink, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.args = args

        self.known_anchor_links = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.train_dict = self.known_anchor_links
        self.number_walks = args.number_walks
        self.format = args.format
        self.walk_length = args.walk_length
        self.window_size = args.window_size
        self.top_k = args.top_k

        self.S = None
        self.source_embedding = None
        self.target_embedding = None
        self.source_after_mapping = None
        self.source_train_nodes = np.array(list(self.train_dict.keys()))

        self.hidden_dim1 = args.hidden_dim1
        self.hidden_dim2 = args.hidden_dim2
        self.seed = args.seed

    def get_alignment_matrix(self):
        return self.S

    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding

    def align(self):
        self.learn_embeddings()

        mapping_model = MappingModel(
                                embedding_dim=self.args.embedding_dim,
                                hidden_dim1=self.hidden_dim1,
                                hidden_dim2=self.hidden_dim2,
                                source_embedding=self.source_embedding,
                                target_embedding=self.target_embedding
                                )

        if self.args.cuda:
            mapping_model = mapping_model.cuda()
        print("Start Unsupervised mapping")
        m_optimizer_us = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()), lr = self.args.unsupervised_lr)
        print("Start Supervised mapping")
        self.mapping_train_(mapping_model, m_optimizer_us, 'us')

        m_optimizer_s = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()), lr = self.args.supervised_lr)
        self.mapping_train_(mapping_model, m_optimizer_s, 's')
        self.source_after_mapping = mapping_model(self.source_embedding, 'val')
        self.S = torch.matmul(self.source_after_mapping, self.target_embedding.t())

        self.S = self.S.detach().cpu().numpy()
        return self.S


    def mapping_train_(self, model, optimizer, mode='s'):
        source_train_nodes = self.source_train_nodes

        batch_size = self.args.batch_size_mapping
        n_iters = len(source_train_nodes)//batch_size
        assert n_iters > 0, "batch_size is too large"
        if(len(source_train_nodes) % batch_size > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        train_dict = self.train_dict
        if mode == 's':
            n_epochs = self.args.supervised_epochs
        else:
            n_epochs = self.args.unsupervised_epochs

        for epoch in range(1, n_epochs+1):
            print("Epoch {0}".format(epoch))
            np.random.shuffle(source_train_nodes)
            for iter in range(n_iters):
                source_batch = source_train_nodes[iter*batch_size:(iter+1)*batch_size]
                target_batch = [train_dict[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.args.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                if mode == 'us':
                    loss = model.unsupervised_loss(source_batch, target_batch)
                else:
                    loss = model.supervised_loss(source_batch, target_batch, alpha=self.args.alpha, k=self.top_k)
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0 and total_steps > 0:
                    print("Iter:", '%03d' %iter,
                          "train_loss=", "{:.4f}".format(loss.item()),
                          "Mode {}".format(mode)
                          )
            
                total_steps += 1


    def learn_embeddings(self):
        print("Start embedding for source nodes, using deepwalk")

        source_embedding_model = DeepWalk(self.source_dataset.G, self.source_dataset.id2idx, self.number_walks, \
                        self.walk_length, self.window_size, self.args.embedding_dim, self.args.num_cores, self.args.embedding_epochs, seed=self.seed)

        self.source_embedding = torch.Tensor(source_embedding_model.get_embedding())
        print("Start embedding for target nodes, using deepwalk")

        target_embedding_model = DeepWalk(self.target_dataset.G, self.target_dataset.id2idx, self.number_walks, \
                        self.walk_length, self.window_size, self.args.embedding_dim, self.args.num_cores, self.args.embedding_epochs, seed=self.seed)

        self.target_embedding = torch.Tensor(target_embedding_model.get_embedding())
        if self.args.cuda:
            self.source_embedding = self.source_embedding.cuda()
            self.target_embedding = self.target_embedding.cuda()

