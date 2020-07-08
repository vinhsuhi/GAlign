from algorithms.network_alignment_model import NetworkAlignmentModel
# from algorithms.DeepLink.mapping_model import MappingModel
from algorithms.CENALP.embedding_model import EmbeddingModel, LinkPredModel

from input.dataset import Dataset
from utils.graph_utils import load_gt
from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch
import networkx as nx

import argparse
import time
import os
import math

from numpy import unravel_index
# import torch.nn.functional as F


class CENALP(NetworkAlignmentModel):
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

        super(CENALP, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.args = args

        self.known_anchor_links = load_gt(args.train_dict, format='dict')
        self.pi = self.known_anchor_links.copy()
        self.cur_iter = 0

        source_deg = self.source_dataset.get_nodes_degrees()
        target_deg = self.target_dataset.get_nodes_degrees()

        self.idx2id_source = {v:k for k,v in self.source_dataset.id2idx.items()}
        self.idx2id_target = {v:k for k,v in self.target_dataset.id2idx.items()}

        self.deg = np.concatenate((source_deg, target_deg))

        if self.source_dataset.features is not None:
            self.sim_attr = self.source_dataset.features.dot(self.target_dataset.features.T)
            self.sim_attr[self.sim_attr < 0] = 0
        else:
            self.sim_attr = np.zeros((len(self.source_dataset.G.nodes()), len(self.target_dataset.G.nodes())))
        

    def get_alignment_matrix(self):
        return self.S
    
    def get_sku(self, nodes, G, rectified_params):
        """
        :params nodes: list of nodes
        """
        sku = [G.degree(node) * rectified_params for node in nodes]
        return sku 


    def dist_two_nodes(self, node1, node2, hop=2):
        nes = len(self.source_dataset.G.edges())
        net = len(self.target_dataset.G.edges())
        nns = len(self.source_dataset.G.nodes())
        nnt = len(self.target_dataset.G.nodes())
        rectified_params_source = np.sqrt((net/nnt) / (nes/nns))
        rectified_params_target = 1 / rectified_params_source

        source_neighbors = [node1]
        target_neighbors = [node2]
        dist = 0
        for _ in range(hop + 1):
            skus = self.get_sku(source_neighbors, self.source_dataset.G, rectified_params_source)
            skut = self.get_sku(target_neighbors, self.target_dataset.G, rectified_params_target)
            dist_i = np.abs(np.log(np.min(skus) + 1) - np.log(np.min(skut) + 1))
            dist_i += np.abs(np.log(np.max(skus) + 1) - np.log(np.max(skut) + 1))
            dist += dist_i
            new_source_neighbors = []
            for ele in source_neighbors:
                new_source_neighbors += self.source_dataset.G.neighbors(ele)
            new_target_neighbors = []
            for ele in target_neighbors:
                new_target_neighbors += self.target_dataset.G.neighbors(ele)
            source_neighbors = new_source_neighbors
            target_neighbors = new_target_neighbors
        return np.exp( -self.args.alpha * dist)


    def get_dist_dict(self):
        dist_dict = {}
        for node in tqdm(self.source_dataset.G.nodes()):
            for target_node in self.target_dataset.G.nodes():
                dist_dict[(node, target_node)] = self.dist_two_nodes(node, target_node, hop=2)
        return dist_dict


    def get_min_max_deg(self, adj, degree):
        deg_matrix = adj * degree
        max_deg = deg_matrix.max(axis=1)
        deg_matrix[deg_matrix == 0] = 1e10
        min_deg = deg_matrix.min(axis=1)
        return min_deg, max_deg


    def compute_2hop_adj(self, adj):
        new_adj = np.zeros_like(adj)
        for i in range(len(new_adj)):
            new_adj[i] = adj[np.where(adj[i] == 1)[0].astype(int)].sum(axis=0)
        new_adj[new_adj>0] = 1
        return new_adj


    def get_dist_matrix(self):
        """
        Degree matrix
        """
        nes = len(self.source_dataset.G.edges())
        net = len(self.target_dataset.G.edges())
        nns = len(self.source_dataset.G.nodes())
        nnt = len(self.target_dataset.G.nodes())
        rectified_params_source = np.sqrt((net/nnt) / (nes/nns))
        rectified_params_target = 1 / rectified_params_source

        source_adj = self.source_dataset.get_adjacency_matrix()
        target_adj = self.target_dataset.get_adjacency_matrix()

        source_degree = source_adj.sum(axis=1) * rectified_params_source
        target_degree = target_adj.sum(axis=1) * rectified_params_target

        source_adj_2hop = self.compute_2hop_adj(source_adj)
        target_adj_2hop = self.compute_2hop_adj(target_adj)

        # we need to define max_deg_neib_source and min_deg_neib_source (target)
        min_deg_neib_source, max_deg_neib_source = self.get_min_max_deg(source_adj, source_degree)
        min_deg_neib_target, max_deg_neib_target = self.get_min_max_deg(target_adj, target_degree)

        min_deg_neib_source2, max_deg_neib_source2 = self.get_min_max_deg(source_adj_2hop, source_degree)
        min_deg_neib_target2, max_deg_neib_target2 = self.get_min_max_deg(target_adj_2hop, target_degree)

        dist = np.zeros((source_adj.shape[0], target_adj.shape[0]))

        for i in tqdm(range(source_adj.shape[0])):
            dist[i] += np.abs(np.log(source_degree[i] + 1) - np.log(target_degree + 1)) * 2
            dist[i] += np.abs(np.log(min_deg_neib_source[i] + 1) - np.log(min_deg_neib_target + 1)) + \
                        np.abs(np.log(max_deg_neib_source[i] + 1) - np.log(max_deg_neib_target + 1))

            dist[i] += np.abs(np.log(min_deg_neib_source2[i] + 1) - np.log(min_deg_neib_target2 + 1)) + \
                        np.abs(np.log(max_deg_neib_source2[i] + 1) - np.log(max_deg_neib_target2 + 1))
        dist =  np.exp( -self.args.alpha * dist)
        return dist


    def dist_to_distdict(self, dist):
        dist_dict = {}
        for node in self.source_dataset.G.nodes():
            for target_node in self.target_dataset.G.nodes():
                dist_dict[(node, target_node)] = dist[self.source_dataset.id2idx[node], self.target_dataset.id2idx[target_node]]
        return dist_dict


    def align(self):
        """
        This is algorithm 3
        """
        # import pickle
        # sourcedt = self.args.source_dataset.split('/')[-3]
        # targetdt = self.args.target_dataset.split('/')[-3]
        # print("Computing distance matrix...")
        # if os.path.exists('./dist-dict/{}_{}.pkl'.format(sourcedt,targetdt)):
        #     with open('./dist-dict/{}_{}.pkl'.format(sourcedt,targetdt),'rb') as f:
        #         self.dist_dict = pickle.load(f)
        # else:
        #     self.dist_dict = self.get_dist_dict()
        #     with open('./dist-dict/{}_{}.pkl'.format(sourcedt,targetdt),'wb') as f:
        #         pickle.dump(self.dist_dict,f)
        self.dist_dict = self.dist_to_distdict(self.get_dist_matrix())
        print("Embedding...")

        # Embedding model
        embedding_model = EmbeddingModel(self.source_dataset.G, self.target_dataset.G, \
                                self.source_dataset.id2idx, self.target_dataset.id2idx, \
                                self.args.num_walks, self.args.walk_len, \
                                self.args.embedding_dim, self.args.switch_prob, \
                                self.dist_dict, self.args.neg_sample_size, self.deg, self.args.cuda, self.pi)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=self.args.learning_rate)

        link_model_source = LinkPredModel(3 * self.args.embedding_dim, 1)
        source_linkpred_optimizer = torch.optim.Adam(link_model_source.parameters(), lr=self.args.learning_rate)
        link_model_target = LinkPredModel(3 * self.args.embedding_dim, 1)
        target_linkpred_optimizer = torch.optim.Adam(link_model_target.parameters(), lr=self.args.learning_rate)

        if self.args.cuda:
            embedding_model = embedding_model.cuda()
            link_model_source = link_model_source.cuda()
            link_model_target = link_model_target.cuda()
        
        models_linkpred = [link_model_source, link_model_target]
        optimizers_linkpred = [source_linkpred_optimizer, target_linkpred_optimizer]

        source_embedding = None
        target_embedding = None
        cur_step = 0
        pairs = 0
        linear_toadd = int(2*min(len(self.source_dataset.G.nodes()),len(self.target_dataset.G.nodes())) / (self.args.num_iteration_epochs * (self.args.num_iteration_epochs + 1)))
        num_pair_toadd = min(15, linear_toadd)
        for num_linear in range(self.args.num_iteration_epochs):
            cur_step += 1
            print("Step: {}".format(cur_step))
            # Step 1: Run a epoch of embedding
            # random walking!
            embedding_model.train()
            source_embedding, target_embedding, pairs = self.learn_embeddings(pairs, embedding_model, optimizer)
            #print("alignment time: {:.4f}".format(time.time() - start_time))

            # Run algorithm 1
            embedding_model.eval()
            start_time = time.time()
            if not self.network_alignment(source_embedding, target_embedding, T=num_pair_toadd):
                break
            num_pair_toadd += linear_toadd
            print("alignment time: {:.4f}".format(time.time() - start_time))

            
            # Run algrithm 2
            if self.args.linkpred_epochs > 0:
                start_time = time.time()
                self.Cross_graph_Linkpred(self.args.threshold, source_embedding, target_embedding, num_sample=self.args.num_sample, \
                    num_epochs_linkpred=self.args.linkpred_epochs, models=models_linkpred, optimizers=optimizers_linkpred)
                print("linkpred time: {:.4f}".format(time.time() - start_time))
        
        self.S = source_embedding.dot(target_embedding.T)
        self.S *= self.sim_attr

        return self.S



    def Cross_graph_Linkpred(self, threshold, source_embedding, target_embedding, num_sample, num_epochs_linkpred, models, optimizers):
        """
        Randomly sample existent links Eext, E'ext
        Randomly sample nonexistent links Emis, E'mis
        """

        source_embedding = torch.FloatTensor(source_embedding)
        target_embedding = torch.FloatTensor(target_embedding)
        if self.args.cuda:
            source_embedding = source_embedding.cuda()
            target_embedding = target_embedding.cuda()

        #################
        # Put your code here... DONE
        source_edges = self.source_dataset.G.edges()
        target_edges = self.target_dataset.G.edges()
        source_non_edges = list(nx.non_edges(self.source_dataset.G))
        target_non_edges = list(nx.non_edges(self.target_dataset.G))
        # Eext, E'ext
        index_Eext_source = np.random.choice(np.arange(len(source_edges)), num_sample)
        index_Eext_target = np.random.choice(np.arange(len(target_edges)), num_sample)
        index_Emis_source = np.random.choice(np.arange(len(source_non_edges)), num_sample)
        index_Emis_target = np.random.choice(np.arange(len(target_non_edges)), num_sample)
        source_edges = np.array(source_edges)
        target_edges = np.array(target_edges)
        source_non_edges = np.array(source_non_edges)
        target_non_edges = np.array(target_non_edges)
        # Eext_source = np.random.choice(source_edges, num_sample)
        Eext_source = np.array(source_edges[index_Eext_source])
        Eext_target = np.array(target_edges[index_Eext_target])
        Emis_source = np.array(source_non_edges[index_Emis_source])
        Emis_target = np.array(target_non_edges[index_Emis_target]) # first wrong
        # Eext_target = np.random.choice(target_edges, num_sample)
        # Emis_source = np.random.choice(source_non_edges, num_sample)
        # Emis_target = np.random.choice(target_non_edges, num_sample)
        Emis_source = np.array([[self.source_dataset.id2idx[Emis_source[i][0]], self.source_dataset.id2idx[Emis_source[i][1]]] for i in range(len(Emis_source))])
        Emis_target = np.array([[self.target_dataset.id2idx[Emis_target[i][0]], self.target_dataset.id2idx[Emis_target[i][1]]] for i in range(len(Emis_target))])
        Eext_source = np.array([[self.source_dataset.id2idx[Eext_source[i][0]], self.source_dataset.id2idx[Eext_source[i][1]]] for i in range(len(Eext_source))])
        Eext_target = np.array([[self.target_dataset.id2idx[Eext_target[i][0]], self.target_dataset.id2idx[Eext_target[i][1]]] for i in range(len(Eext_target))])

        label_source = np.concatenate((np.ones(len(Eext_source)), np.zeros(len(Emis_source))))
        label_target = np.concatenate((np.ones(len(Eext_target)), np.zeros(len(Emis_target))))
        Esource = np.concatenate((Eext_source, Emis_source), axis=0)
        Etarget = np.concatenate((Eext_target, Emis_target), axis=0)
        Esource = torch.LongTensor(Esource)
        Etarget = torch.LongTensor(Etarget)
        label_source = torch.LongTensor(label_source)
        label_target = torch.LongTensor(label_target)
        if self.args.cuda:
            Esource = Esource.cuda()
            Etarget = Etarget.cuda()
            label_source = label_source.cuda()
            label_target = label_target.cuda()

        labels = [label_source, label_target]
        E = [Esource, Etarget]
        embeddings = [source_embedding, target_embedding]
        #################
        
        """
        Create Link Prediction classifiter model W and W' ||| Nooooo!!!!
        """
        #################
        # Put your code here... DONE!
        
        #################

        """
        Train linkpred
        """

        #################
        # Put your code here...
        for epoch in range(num_epochs_linkpred):
            print("Linkpred Epoch: {}".format(epoch))
            for i in range(2):
                model = models[i]
                optimizer = optimizers[i]
                optimizer.zero_grad()
                # Eq. 14a
                batch_edge = E[i]
                head_batch_nodes = embeddings[i][batch_edge[:, 0]]
                tail_batch_nodes = embeddings[i][batch_edge[:, 1]]

                cat = torch.cat((head_batch_nodes, head_batch_nodes, head_batch_nodes * tail_batch_nodes), dim=1)
                output = model(cat)
                loss = 0
                constant = 1e-10
                for k in range(len(output)):
                    loss -= labels[i][k] * torch.log(output[k][0] + constant) + (1 - labels[i][k]) * torch.log(1 - output[k][0] + constant)
                loss /= len(output)
                print("Loss: {:.4f}".format(loss.item()))
                loss.backward()
                optimizer.step()
            
        #################

        """
        Construct edge test in subgraph of seed set Etest, E'test [all possible pairs]
        """
        #################
        # Put your code here...
        source_seed_set = list(self.pi.keys())
        target_seed_set = list(self.pi.values())

        for i in range(len(source_seed_set) - 1):
            head_nodes = source_seed_set[i]
            head_nodet = target_seed_set[i]

            embedding_head_nodes = source_embedding[self.source_dataset.id2idx[head_nodes]]
            embedding_head_nodet = target_embedding[self.target_dataset.id2idx[head_nodet]]
            for j in range(i, len(source_seed_set)):
                tail_nodes = source_seed_set[j]
                tail_nodet = target_seed_set[j]

                embedding_tail_nodes = source_embedding[self.source_dataset.id2idx[tail_nodes]]
                embedding_tail_nodet = target_embedding[self.target_dataset.id2idx[tail_nodet]]

                cat_source = torch.cat((embedding_head_nodes, embedding_head_nodes, embedding_head_nodes * embedding_tail_nodes))
                cat_target = torch.cat((embedding_head_nodet, embedding_head_nodet, embedding_head_nodet * embedding_tail_nodet))

                output_source = models[0](cat_source)
                output_target = models[1](cat_target)

                if output_source > threshold and output_target > threshold:
                    self.source_dataset.G.add_edge(head_nodes, tail_nodes)
                    self.target_dataset.G.add_edge(head_nodet, tail_nodet)

        


    def network_alignment(self, source_embedding, target_embedding, T=10):
        """
        This is algorithm 1
        """
        # S and S'
        source_seed_set = set(self.pi.keys())
        target_seed_set = set(self.pi.values())


        ##### sim_emb (Eq. 9)
        sim_emb = source_embedding.dot(target_embedding.T)
        sim_emb[sim_emb < 0] = 0
        

        # sim_jc
        sim_jc = np.zeros_like(sim_emb)
        # Create NG1(S), one hop neighbors of S
        N_source = set()
        for node in source_seed_set:
            node_neighbors = self.source_dataset.G.neighbors(node)
            for neib in node_neighbors:
                # if neib not in source_seed_set:
                N_source.add(neib)

        # Create NG'1(S'), one hop neighbors of S'
        N_target = set()
        for node in target_seed_set:
            node_neighbors = self.target_dataset.G.neighbors(node)
            for neib in node_neighbors:
                # if neib not in target_seed_set:
                N_target.add(neib)

        
        N_source = N_source.difference(source_seed_set)
        N_target = N_target.difference(target_seed_set)
        
        # if one hop neibbor dont have any element, then two possible cases:
        # 1. it's over! => Return False, can't find any pair more
        # 2. seed set is empty => sim_jc = 1


        if len(N_source) == 0 or len(N_target) == 0:
            if len(source_seed_set) > 0:
                return False
        # update sim_jc as Eq. 10
        for nodes in N_source:
            neighbor_source = self.source_dataset.G.neighbors(nodes)
            # care_neighbor_source = [neib for neib in neighbor_source if neib in source_seed_set]
            care_neighbor_source = set(neighbor_source).intersection(source_seed_set)
            care_neighbor_source_to_target = set([self.pi[ele] for ele in care_neighbor_source])
            for nodet in N_target:
                neighbor_target = self.target_dataset.G.neighbors(nodet)
                # care_neighbor_target = set([neib for neib in neighbor_target if neib in target_seed_set])
                care_neighbor_target = set(neighbor_target).intersection(target_seed_set)
                sim_jc[self.source_dataset.id2idx[nodes], self.target_dataset.id2idx[nodet]] = len(care_neighbor_source_to_target.intersection(care_neighbor_target)) / (len(care_neighbor_source_to_target.union(care_neighbor_target)))

        if sim_jc.sum() == 0:
            sim_jc += 1


        ###### sim_graph as Eq. 11
        sim_graph = sim_emb * sim_jc


        ###### sim_attr as Eq. 12
        


        ###### sim as Eq. 13
        sim = sim_graph * self.sim_attr

        num_pair_before = len(self.pi)
        while T > 0:
            T -= 1
            argmaxx = unravel_index(sim.argmax(), sim.shape)
            sim[argmaxx[0], argmaxx[1]] = 0
            self.pi[self.idx2id_source[argmaxx[0]]] = self.idx2id_target[argmaxx[1]]
        num_pair_after = len(self.pi)

        if num_pair_before == num_pair_after:
            return False
        return True




    def learn_embeddings(self, pairs, embedding_model, optimizer):
        """
        Todo: Return source embedding and target embedding
        """
        print("Start embedding")
        if self.cur_iter % self.args.walk_every == 0 :
            start_time = time.time()
            pairs = embedding_model.run_random_walks(self.pi)
            print("walking time: {:.4f}".format(time.time() - start_time))
        self.cur_iter += 1
        batch_size = min(int(len(pairs) / 4), 512)
        n_iters = len(pairs) // batch_size
        assert n_iters > 0, "batch_size is too large!"
        if(len(pairs) % batch_size > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        print("Number of pairs: {}".format(len(pairs)))
        # embedding? - with 1 epochs

        for epoch in range(1):
            start_time = time.time()
            print("Epochs: {}".format(epoch))
            np.random.shuffle(pairs)
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(pairs[iter*batch_size:(iter+1)*batch_size])
                if self.args.cuda:
                    batch_edges = batch_edges.cuda()
                start_time = time.time()
                optimizer.zero_grad()
                loss = embedding_model.loss(batch_edges[:,0], batch_edges[:, 1])
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0:
                    print("Iter: {}, train_loss: {:.4f}, time: {:.4f}".format(iter, loss.item(), time.time() - start_time))
                total_steps += 1
            print("Time per embedding epoch: {:.4f}".format(time.time() - start_time))
        embeddings = embedding_model.node_embedding.weight.detach().cpu().numpy()
        
        source_embedding = embeddings[:len(self.source_dataset.G.nodes())]
        target_embedding = embeddings[len(self.source_dataset.G.nodes()):]

        return source_embedding, target_embedding, pairs
