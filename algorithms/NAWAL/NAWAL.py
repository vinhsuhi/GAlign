from algorithms.network_alignment_model import NetworkAlignmentModel
from algorithms.NAWAL.embedding_model import PaleEmbedding
from algorithms.map_architechtures import PaleMappingLinear, Discriminator, Mapping, EncoderLinear, EncoderMLP, DecoderLinear, DecoderMLP, PaleMappingMlp, MappingModel
from evaluation.metrics import get_statistics

from input.dataset import Dataset
from utils.graph_utils import load_gt
from algorithms.NAWAL.utils import to_word2vec_format, read_embedding_from_file, get_optimizer, build_dictionary
import torch.nn.functional as F
from torch import Tensor as torch_tensor
import scipy

import torch
import numpy as np

import argparse
import os
import time
from algorithms.DeepLink.embedding_model import DeepWalk



class NAWAL(NetworkAlignmentModel):
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

        super(NAWAL, self).__init__(source_dataset, target_dataset)
        # dataset
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        # embedding_params
        self.args = args

        self.pale_train_anchors = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.train_dict = self.pale_train_anchors
        self.nawal_test_anchors = load_gt(args.test_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.test_dict = self.nawal_test_anchors
        self.source_train_nodes = np.array(list(self.pale_train_anchors.keys()))

        # nawal_mapping_params
        self.decrease_lr = False
        # if use auto_encoder mapping
        self.n_refinement = 5

        self.source_embedding = None
        self.target_embedding = None
        self.mean_cosine = -1
        self.best_valid_metric = -1
        self.best_W = None
        self.encoder = None
        self.decoder = None


    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding

    def learn_UAGA(self):
        print("Start embedding for source nodes, using deepwalk")

        source_embedding_model = DeepWalk(self.source_dataset.G, self.source_dataset.id2idx, 80, \
                        40, 5, 32, 16, 5, seed = 1)

        self.source_embedding = torch.Tensor(source_embedding_model.get_embedding())
        print("Start embedding for target nodes, using deepwalk")

        target_embedding_model = DeepWalk(self.target_dataset.G, self.target_dataset.id2idx, 80, \
                        40, 5, 32, 16, 5, seed = 1)

        self.target_embedding = torch.Tensor(target_embedding_model.get_embedding())
        if self.args.cuda:
            self.source_embedding = self.source_embedding.cuda()
            self.target_embedding = self.target_embedding.cuda()


    def align(self):
        if self.args.load_emb:
            self.source_embedding = read_embedding_from_file(self.source_dataset.id2idx, \
                "algorithms/NAWAL/embeddings/{}_source".format(self.args.embedding_name), self.args.embedding_dim, self.args.cuda)
            self.target_embedding = read_embedding_from_file(self.target_dataset.id2idx, \
                "algorithms/NAWAL/embeddings/{}_target".format(self.args.embedding_name), self.args.embedding_dim, self.args.cuda)
        else:
            if self.args.UAGA_mode:
                self.learn_UAGA()
            else:
                self.learn_embeddings()
        
        if self.args.mapper == "nawal":
            self.S = self.nawal_mapping()
        elif self.args.mapper == "deeplink":
            self.S = self.deeplink_mapping()
        elif self.args.mapper == "linear":
            self.S = self.pale_mapping('linear')
        elif self.args.mapper == "mlp":
            self.S = self.pale_mapping('mlp')
        else:
            s_deep = self.deeplink_mapping()
            s_linear = self.pale_mapping('linear')
            s_mlp = self.pale_mapping('mlp')
            return s_deep, s_linear, s_mlp

        return self.S


    def deeplink_mapping(self):
        mapping_model = MappingModel(
                                embedding_dim=self.args.embedding_dim,
                                hidden_dim1=self.args.hidden_dim1,
                                hidden_dim2=self.args.hidden_dim2,
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
        S = torch.matmul(self.source_after_mapping, self.target_embedding.t())

        S = S.detach().cpu().numpy()
        return S


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
                    loss = model.supervised_loss(source_batch, target_batch, alpha=self.args.alpha, k=self.args.top_k)
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0 and total_steps > 0:
                    print("Iter:", '%03d' %iter,
                          "train_loss=", "{:.4f}".format(loss.item()),
                          "Mode {}".format(mode)
                          )
                total_steps += 1


    def procrustes(self, dico):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.source_embedding[dico[:, 0]]
        B = self.target_embedding[dico[:, 1]]
        W = self.mapping.layer.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, _, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))


    def save_best(self):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        if self.mean_cosine > self.best_valid_metric:
            #print("Mean cosine is larger thann the best: {} > {}".format(self.mean_cosine, self.best_valid_metric))
            # new best mapping
            self.best_valid_metric = self.mean_cosine
            self.best_W = self.mapping.layer.weight.detach().cpu().numpy()
        else:
            #print("Mean cosine is smaller than the best: {} < {}".format(self.mean_cosine, self.best_valid_metric))
            pass
        
    
    def dist_mean_cosine(self):
        """
        Mean-cosine model selection criterion.
        """
        # get normalized embeddings
        src_emb = self.mapping(self.source_embedding).data
        tgt_emb = self.target_embedding.data
        # build dictionary
        dico = build_dictionary(src_emb, tgt_emb)
        
        if dico is None:
            mean_cosine = -1e9
        else:
            mean_cosine = (src_emb[dico[:, 0]] * tgt_emb[dico[:, 1]]).sum(1).mean()
        
        self.mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
        return self.mean_cosine


    def reload_best(self):
        """
        Reload the best mapping.
        """
        to_reload = torch.from_numpy(self.best_W)
        W = self.mapping.layer.weight.data
        W.copy_(to_reload.type_as(W))


    def calculate_simi_matrix(self, mapping_model, save=False):
        source_after_mapping = mapping_model(self.source_embedding)
        source_after_mapping = F.normalize(source_after_mapping, dim=1)
        self.target_embedding = F.normalize(self.target_embedding, dim=1)
        S = torch.matmul(source_after_mapping, self.target_embedding.t())
        S = S.detach().cpu().numpy()
        return S       


    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.args.nawal_mapping_batch_size

        src_ids = torch.LongTensor(bs).random_(len(self.source_embedding))
        tgt_ids = torch.LongTensor(bs).random_(len(self.target_embedding))

        if self.args.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()
        
        with torch.no_grad():
            src_emb = self.source_embedding[src_ids]
            tgt_emb = self.target_embedding[tgt_ids]
        if volatile:
            with torch.no_grad():
                src_emb = self.mapping(src_emb)
                tgt_emb = tgt_emb
        else:
            src_emb = self.mapping(src_emb)
            tgt_emb = tgt_emb
        # input / target
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.args.dis_smooth
        y[bs:] = self.args.dis_smooth
        if self.args.cuda:
            y = y.cuda()
        return x, y


    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.discriminator.train()

        # loss
        x, y = self.get_dis_xy(volatile=True)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, y)
        stats['DIS_COSTS'].append(loss.item())

        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (discriminator)")
            exit()

        # optim
        self.dis_optimizer.zero_grad()
        loss.backward()
        self.dis_optimizer.step()


    def mapping_step(self):
        """
        Fooling discriminator training step.
        """
        self.discriminator.eval()

        # loss
        x, y = self.get_dis_xy(volatile=False)
        preds = self.discriminator(x)
        loss = F.binary_cross_entropy(preds, 1 - y)
        
        # check NaN
        if (loss != loss).data.any():
            print("NaN detected (fool discriminator)")
            exit()

        # optim
        self.map_optimizer.zero_grad()
        loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.args.nawal_mapping_batch_size


    def update_lr(self):
        """
        Update learning rate when using SGD.
        """
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.args.min_lr, old_lr * self.args.lr_decay)
        if new_lr < old_lr:
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.args.lr_shrink < 1 and self.mean_cosine >= -1e7:
            if self.mean_cosine < self.best_valid_metric:
                if self.decrease_lr:
                    old_lr = self.map_optimizer.param_groups[0]['lr']
                    self.map_optimizer.param_groups[0]['lr'] *= self.args.lr_shrink
                self.decrease_lr = True


    def nawal_mapping(self):
        #print("Start nawal mapping")
        self.mapping = Mapping(self.args.embedding_dim)
        self.discriminator = Discriminator(self.args.embedding_dim, self.args.dis_layers, self.args.dis_hid_dim, self.args.dis_dropout, self.args.dis_input_dropout)
        
        optim_fn, optim_params = get_optimizer(self.args.map_optimizer)
        self.map_optimizer = optim_fn(self.mapping.parameters(), **optim_params)
        optim_fn, optim_params = get_optimizer(self.args.dis_optimizer)
        self.dis_optimizer = optim_fn(self.discriminator.parameters(), **optim_params)

        if self.args.cuda:
            self.mapping = self.mapping.cuda()
            self.discriminator = self.discriminator.cuda()
        nawal_map_epoch_times = []
        for n_epoch in range(self.args.nawal_mapping_epochs):
            print('Starting adversarial training epoch %i...' % n_epoch)
            tic = time.time()
            n_nodes_proc = 0
            stats = {'DIS_COSTS': []}

            for n_iter in range(0, self.args.nawal_mapping_epoch_size, self.args.nawal_mapping_batch_size):
                # discriminator training
                for _ in range(self.args.dis_steps):
                    self.dis_step(stats)

                # mapping training (discriminator fooling)
                n_nodes_proc += self.mapping_step()

                # log stats
                if n_iter % 500 == 0:
                    stats_str = [('DIS_COSTS', 'Discriminator loss')]
                    stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                                for k, v in stats_str if len(stats[k]) > 0]
                    stats_log.append('%i samples/s' % int(n_nodes_proc / (time.time() - tic)))
                    print(('%06i - ' % n_iter) + ' - '.join(stats_log))

                    # reset
                    tic = time.time()
                    n_nodes_proc = 0
                    for k, _ in stats_str:
                        del stats[k][:]
            # embeddings / discriminator evaluation
            self.dist_mean_cosine()

            # JSON log / save best model / end of epoch
            self.save_best()
            
            print('End of epoch %i.\n\n' % n_epoch)
            nawal_map_epoch_times.append(time.time() - tic)
            # update the learning rate (stop if too small)
            self.update_lr()

        self.reload_best()
        self.S = self.calculate_simi_matrix(self.mapping.eval()) 
        # print("NAWAL before refining")
        groundtruth_matrix = load_gt(self.args.test_dict, self.source_dataset.id2idx, self.target_dataset.id2idx)
        groundtruth_dict = load_gt(self.args.test_dict, self.source_dataset.id2idx, self.target_dataset.id2idx, 'dict')
        self.nawal_before_refine_acc = get_statistics(self.S, groundtruth_dict, groundtruth_matrix)
        # print("Accuracy: {}".format(acc))

        self.mapping.train()
        nawal_refine_epoch_times = []
        # training loop
        for n_iter in range(self.n_refinement):
            tic = time.time()
            # build a dictionary from aligned embeddings
            src_emb = self.mapping(self.source_embedding).data
            tgt_emb = self.target_embedding
            dico = build_dictionary(src_emb, tgt_emb, p_keep=0.45)
            # apply the Procrustes solution
            self.procrustes(dico)
            self.dist_mean_cosine()
            self.save_best()
            nawal_refine_epoch_times.append(time.time() - tic)
        self.reload_best()
        
        S = self.calculate_simi_matrix(self.mapping.eval(), save=True)
        # print("Nawal after refining")
        groundtruth_matrix = load_gt(self.args.test_dict, self.source_dataset.id2idx, self.target_dataset.id2idx)
        groundtruth_dict = load_gt(self.args.test_dict, self.source_dataset.id2idx, self.target_dataset.id2idx, 'dict')
        self.nawal_after_refine_acc = get_statistics(S, groundtruth_dict, groundtruth_matrix)
        self.log()
        print("NAWAL average map epoch time: {:.4f}".format(np.mean(nawal_map_epoch_times)))
        print("NAWAL average refine epoch time: {:.4f}".format(np.mean(nawal_refine_epoch_times)))
        print("NAWAL average emb epoch time: {:.4f}".format(np.mean(self.epoch_times)))
        return S


    def log(self):
        print("NAWAL_BEFORE_REFINING {}".format(self.nawal_before_refine_acc))
        print("NAWAL_AFTER_REFINING {}".format(self.nawal_after_refine_acc))


    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.args.map_beta > 0:
            W = self.mapping.layer.weight.data
            beta = self.args.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))


    def pale_mapping(self, mapper):
        if mapper == "linear":
            print('Using linear mapping')
            mapping_model = PaleMappingLinear(
                                        embedding_dim=self.args.embedding_dim,
                                        source_embedding=self.source_embedding,
                                        target_embedding=self.target_embedding,
                                        )
        else:
            print('Using mlp mapping')
            mapping_model = PaleMappingMlp(
                                        embedding_dim=self.args.embedding_dim,
                                        source_embedding=self.source_embedding,
                                        target_embedding=self.target_embedding,
                                        )

        if self.args.cuda:
            mapping_model = mapping_model.cuda()

        mapping_model.train()

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, mapping_model.parameters()), lr=self.args.pale_map_lr)
        n_iters = len(self.source_train_nodes) // self.args.pale_map_batchsize
        assert n_iters > 0, "batch_size is too large"
        if(len(self.source_train_nodes) % self.args.pale_map_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.args.pale_map_epochs
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start = time.time()
            print('Epochs: ', epoch)
            np.random.shuffle(self.source_train_nodes)
            for iter in range(n_iters):
                source_batch = self.source_train_nodes[iter*self.args.pale_map_batchsize:(iter+1)*self.args.pale_map_batchsize]
                target_batch = [self.pale_train_anchors[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.args.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                start_time = time.time()
                loss = mapping_model.loss(source_batch, target_batch)
                loss.backward()
                optimizer.step()
                
                if total_steps % print_every == 0 and total_steps > 0:
                    print("PALE_MAPPING: Iter:", '%03d' %iter,
                          "train_loss=", "{:.5f}".format(loss.item()),
                          "time", "{:.5f}".format(time.time()-start_time)
                          )
                
                total_steps += 1
            self.mapping_epoch_time = time.time() - start
        S = self.calculate_simi_matrix(mapping_model)
        return S


    def learn_embeddings(self):
        num_source_nodes = len(self.source_dataset.G.nodes())
        source_deg = self.source_dataset.get_nodes_degrees()
        source_edges = self.source_dataset.get_edges()

        num_target_nodes = len(self.target_dataset.G.nodes())
        target_deg = self.target_dataset.get_nodes_degrees()
        target_edges = self.target_dataset.get_edges()

        self.source_embedding = self.learn_embedding(num_source_nodes, source_deg, source_edges, "SOURCE_GRAPH") #, 's')
        self.target_embedding = self.learn_embedding(num_target_nodes, target_deg, target_edges, "TARGET_GRAPH") #, 't')


    def learn_embedding(self, num_nodes, deg, edges, graph_name):
        embedding_model = PaleEmbedding(
                                        n_nodes = num_nodes,
                                        embedding_dim = self.args.embedding_dim,
                                        deg= deg,
                                        neg_sample_size = self.args.neg_sample_size,
                                        cuda = self.args.cuda,
                                        )
        if self.args.cuda:
            embedding_model = embedding_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=self.args.emb_lr)
        embedding = self.train_embedding(embedding_model, edges, optimizer, graph_name)

        return embedding


    def train_embedding(self, embedding_model, edges, optimizer, graph_name='SOURCE_GRAPH'):
        n_iters = len(edges) // self.args.batch_size_embedding
        assert n_iters > 0, "batch_size is too large!"
        if(len(edges) % self.args.batch_size_embedding > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.args.embedding_epochs
        self.epoch_times = []
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start_epoch_times = time.time()
            print("Epoch {0}".format(epoch))
            np.random.shuffle(edges)
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(edges[iter*self.args.batch_size_embedding:(iter+1)*self.args.batch_size_embedding])
                if self.args.cuda:
                    batch_edges = batch_edges.cuda()
                start_time = time.time()
                optimizer.zero_grad()
                loss, _, _ = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
                loss.backward()
                optimizer.step()
                # if total_steps % print_every == 0:
                #     print("EMBEDDING {} Iter:".format(graph_name), '%03d' %iter,
                #               "train_loss=", "{:.5f}".format(loss.item()),
                #               "epoch time", "{:.5f}".format(time.time()-start_time)
                #           )
                total_steps += 1
            self.epoch_times.append(time.time() - start_epoch_times)
        
        print("Average Epoch time: {:.4f}".format(np.mean(self.epoch_times)))    
        embedding = embedding_model.get_embedding()
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.args.cuda:
            embedding = embedding.cuda()

        return embedding



