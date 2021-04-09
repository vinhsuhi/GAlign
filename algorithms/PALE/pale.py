from algorithms.network_alignment_model import NetworkAlignmentModel
from tqdm import tqdm
from algorithms.PALE.embedding_model import PaleEmbedding
from algorithms.PALE.mapping_model import PaleMappingLinear, PaleMappingMlp
from input.dataset import Dataset
from utils.graph_utils import load_gt

import torch
import numpy as np

import argparse
import os
import time



class PALE(NetworkAlignmentModel):
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

        super(PALE, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.source_path = args.source_dataset

        self.emb_batchsize = args.batch_size_embedding
        self.map_batchsize = args.batch_size_mapping
        self.emb_lr = args.learning_rate1
        self.cuda = args.cuda
        self.neg_sample_size = args.neg_sample_size
        self.embedding_dim = args.embedding_dim
        self.emb_epochs = args.embedding_epochs
        self.map_epochs = args.mapping_epochs
        self.mapping_model = args.mapping_model
        self.map_act = args.activate_function
        self.map_lr = args.learning_rate2
        self.embedding_name = args.embedding_name
        self.args = args

        self.gt_train = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')

        self.S = None
        self.source_embedding = None
        self.target_embedding = None
        self.source_after_mapping = None
        self.source_train_nodes = np.array(list(self.gt_train.keys()))

    def get_alignment_matrix(self):
        return self.S

    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding
    
    def align(self):
        self.learn_embeddings()

        self.to_word2vec_format(self.source_embedding, self.source_dataset.G.nodes(), 'algorithms/PALE/embeddings', self.embedding_name + "_source", \
            self.embedding_dim, self.source_dataset.id2idx)

        self.to_word2vec_format(self.target_embedding, self.target_dataset.G.nodes(), 'algorithms/PALE/embeddings', self.embedding_name + "_target", \
            self.embedding_dim, self.target_dataset.id2idx)

        if self.mapping_model == 'linear':
            print("Use linear mapping")
            mapping_model = PaleMappingLinear(
                                        embedding_dim=self.embedding_dim,
                                        source_embedding=self.source_embedding,
                                        target_embedding=self.target_embedding,
                                        )
        else:
            print("Use Mpl mapping")
            mapping_model = PaleMappingMlp(
                                        embedding_dim=self.embedding_dim,
                                        source_embedding=self.source_embedding,
                                        target_embedding=self.target_embedding,
                                        activate_function=self.map_act,
                                        )
        if self.cuda:
            mapping_model = mapping_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, mapping_model.parameters()), lr=self.map_lr)
        n_iters = len(self.source_train_nodes) // self.map_batchsize
        assert n_iters > 0, "batch_size is too large"
        if(len(self.source_train_nodes) % self.map_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.map_epochs
        for epoch in range(1, n_epochs + 1):

            # for time evaluate
            start = time.time()

            print('Epochs: ', epoch)
            np.random.shuffle(self.source_train_nodes)
            for iter in range(n_iters):
                source_batch = self.source_train_nodes[iter*self.map_batchsize:(iter+1)*self.map_batchsize]
                target_batch = [self.gt_train[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                start_time = time.time()
                loss = mapping_model.loss(source_batch, target_batch)
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0 and total_steps > 0:
                    print("Iter:", '%03d' %iter,
                          "train_loss=", "{:.5f}".format(loss.item()),
                          "time", "{:.5f}".format(time.time()-start_time)
                          )
                total_steps += 1
            # for time evaluate
            self.mapping_epoch_time = time.time() - start

        self.source_after_mapping = mapping_model(self.source_embedding)

        self.S = torch.matmul(self.source_after_mapping, self.target_embedding.t())
        self.S = self.S.detach().cpu().numpy()
        np.save("pale_S{}.npy".format(self.embedding_name), self.S)
        return self.S


    def to_word2vec_format(self, val_embeddings, nodes, out_dir, filename, dim, id2idx, pref=""):
        val_embeddings = val_embeddings.cpu().detach().numpy()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open("{0}/{1}".format(out_dir, filename), 'w') as f_out:
            f_out.write("%s %s\n"%(len(nodes), dim))
            for node in nodes:
                txt_vector = ["%s" % val_embeddings[int(id2idx[node])][j] for j in range(dim)]
                f_out.write("%s%s %s\n" % (pref, node, " ".join(txt_vector)))
            f_out.close()
        print("emb has been saved to: {0}/{1}".format(out_dir, filename))



    def check_edge_in_edges(self, edge, edges):
        for e in edges:
            if np.array_equal(edge, e):
                return True
        return False


    def extend_edge(self, source_edges, target_edges):
        for edge in source_edges:
            if edge[0] in self.gt_train.keys():
                if edge[1] in self.gt_train.keys():
                    if not self.check_edge_in_edges(np.array([self.gt_train[edge[0]], self.gt_train[edge[1]]]), target_edges):
                        target_edges = np.concatenate((target_edges, np.array(([[self.gt_train[edge[0]], self.gt_train[edge[1]]]]))), axis=0)
                        target_edges = np.concatenate((target_edges, np.array(([[self.gt_train[edge[1]], self.gt_train[edge[0]]]]))), axis=0)

        inverse_gt_train = {v:k for k, v in self.gt_train.items()}
        for edge in target_edges:
            if edge[0] in self.gt_train.values():
                if edge[1] in self.gt_train.values():
                    if not self.check_edge_in_edges(np.array([inverse_gt_train[edge[0]], inverse_gt_train[edge[1]]]), source_edges):
                        source_edges = np.concatenate((source_edges, np.array(([[inverse_gt_train[edge[0]], inverse_gt_train[edge[1]]]]))), axis=0)
                        source_edges = np.concatenate((source_edges, np.array(([[inverse_gt_train[edge[1]], inverse_gt_train[edge[0]]]]))), axis=0)
        return source_edges, target_edges

    
    def gen_neigbor_dict(self, edges):
        neib_dict = dict()
        for i in range(len(edges)):
            source, target = edges[i, 0], edges[i, 1]
            if source not in neib_dict:
                neib_dict[source] = set([target])
            else:
                neib_dict[source].add(target)

            if target not in neib_dict:
                neib_dict[target] = set([source])
            else:
                neib_dict[target].add(source)
        return neib_dict

    
    def run_walks(self, neib_dict):
        neib_dict = {key: list(value) for key, value in neib_dict.items()}
        walks = []
        # new_edges = []
        print("Random walks...")
        for key, value in tqdm(neib_dict.items()):
            cur_node = key
            for iter in range(self.args.num_walks):
                walk = [cur_node]
                success = False
                count = 1
                while not success:
                    # try ten times
                    for i in range(10):
                        next_node = np.random.choice(neib_dict[key])
                        if next_node in walk:
                            continue 
                        break
                    if next_node in walk:
                        break
                    walk.append(next_node)
                    cur_node = next_node 
                    count += 1
                    if count == self.args.walk_len:
                        success = True 
                if not success:
                    continue 
                walks.append(walk)
        return np.array(walks)





    def learn_embeddings(self):
        num_source_nodes = len(self.source_dataset.G.nodes())
        source_deg = self.source_dataset.get_nodes_degrees()
        source_edges = self.source_dataset.get_edges()

        num_target_nodes = len(self.target_dataset.G.nodes())
        target_deg = self.target_dataset.get_nodes_degrees()
        target_edges = self.target_dataset.get_edges()

        neibor_dict1 = self.gen_neigbor_dict(source_edges)
        neibor_dict2 = self.gen_neigbor_dict(target_edges)

        self.walks1 = self.run_walks(neibor_dict1)
        self.walks2 = self.run_walks(neibor_dict2)

        #source_edges, target_edges = self.extend_edge(source_edges, target_edges)
        
        print("Done extend edges")
        self.source_embedding = self.learn_embedding(num_source_nodes, source_deg, source_edges, self.walks1) #, 's')
        self.target_embedding = self.learn_embedding(num_target_nodes, target_deg, target_edges, self.walks2) #, 't')


    def learn_embedding(self, num_nodes, deg, edges, walks):
        embedding_model = PaleEmbedding(
                                        n_nodes = num_nodes,
                                        embedding_dim = self.embedding_dim,
                                        deg= deg,
                                        neg_sample_size = self.neg_sample_size,
                                        cuda = self.cuda,
                                        )
        if self.cuda:
            embedding_model = embedding_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=self.emb_lr)
        embedding = self.train_embedding(embedding_model, edges, optimizer, walks)

        return embedding


    def train_embedding(self, embedding_model, edges, optimizer, walks):
        n_iters = len(edges) // self.emb_batchsize
        assert n_iters > 0, "batch_size is too large!"
        if(len(edges) % self.emb_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        walk_batch_size = len(walks) // n_iters
        total_steps = 0
        n_epochs = self.emb_epochs
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start = time.time()

            print("Epoch {0}".format(epoch))
            np.random.shuffle(edges)
            np.random.shuffle(walks)
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(edges[iter*self.emb_batchsize:(iter+1)*self.emb_batchsize])
                batch_walks = torch.LongTensor(walks[iter*walk_batch_size:(iter+1)*walk_batch_size])
                if self.cuda:
                    batch_edges = batch_edges.cuda()
                    batch_walks = batch_walks.cuda()
                
                start_time = time.time()
                optimizer.zero_grad()
                loss, loss0, loss1 = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0:
                    print("Iter:", '%03d' %iter,
                              "train_loss=", "{:.5f}".format(loss.item()),
                              "time", "{:.5f}".format(time.time()-start_time)
                          )
                total_steps += 1
            
            # for time evaluate
            self.embedding_epoch_time = time.time() - start
            
        embedding = embedding_model.get_embedding()
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.cuda:
            embedding = embedding.cuda()

        return embedding

