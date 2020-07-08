from gensim.models import Word2Vec
import numpy as np

from algorithms.CENALP.loss import EmbeddingLossFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled


class EmbeddingModel(nn.Module):
    def __init__(self, Gs, Gt, id2idxs, id2idxt, num_walks=10, walk_len=10, \
                embedding_dim=64, switch_prob=0.5, dist_dict={}, neg_sample_size=10, deg=[], cuda=True, test_dict={}):

        super(EmbeddingModel, self).__init__()
        
        """
        Parameters
        ----------
        G: networkx Graph
            Graph
        id2idx: dictionary
            dictionary of keys are ids of nodes and values are index of nodes
        num_walks: int
            number of walks per node
        walk_len: int
            length of each walk
        window_size: int
            size of windows in skip gram model
        embedding_dim: int
            number of embedding dimensions
        num_cores: int
            number of core when train embedding
        num_epochs: int
            number of epochs in embedding
        seed: seed of gensim
        """
        # self.num_cores = num_cores # CLEAN
        # self.window_size = window_size # CLEAN
        # self.num_epochs = num_epochs # CLEAN
        # self.seed = seed # CLEAN
        


        self.Gs = Gs
        self.Gt = Gt 

        self.num_walks = num_walks
        self.walk_len = walk_len
        self.id2idxs = id2idxs
        self.id2idxt = id2idxt
        self.embedding_dim = embedding_dim
        self.switch_prob = switch_prob # does not change
        self.dist_dict = dist_dict # does not change over training

        self.link_pred_layer = EmbeddingLossFunctions()

        self.node_embedding = nn.Embedding(len(Gs.nodes()) + len(Gt.nodes()), embedding_dim)
        self.neg_sample_size = neg_sample_size
        self.deg = deg
        self.use_cuda = cuda
        self.test_dict = test_dict
        self.inv_test_dict = {v:k for k, v in test_dict.items()}
        

    def loss(self, nodes, neighbor_nodes):
        batch_output, neighbor_output, neg_output = self.forward(nodes, neighbor_nodes)
        batch_size = batch_output.shape[0]
        loss = self.link_pred_layer.loss(batch_output, neighbor_output, neg_output)
        loss = loss/batch_size
        return loss


    def forward(self, nodes, neighbor_nodes=None):
        node_output = self.node_embedding(nodes)
        node_output = F.normalize(node_output, dim=1)

        if neighbor_nodes is not None:
            neg = fixed_unigram_candidate_sampler(
                num_sampled=self.neg_sample_size,
                unique=False,
                range_max=len(self.deg),
                distortion=0.75,
                unigrams=self.deg
                )

            neg = torch.LongTensor(neg)
            
            if self.use_cuda:
                neg = neg.cuda()
            neighbor_output = self.node_embedding(neighbor_nodes)
            neg_output = self.node_embedding(neg)
            # normalize
            neighbor_output = F.normalize(neighbor_output, dim=1)
            neg_output = F.normalize(neg_output, dim=1)

            return node_output, neighbor_output, neg_output

        return node_output


    def run_random_walks(self, test_dict):
        print("Random walk process")
        pairs = []
        nets = [self.Gs, self.Gt]
        suffices = [0, len(self.Gs.nodes())]
        id2idices = [self.id2idxs, self.id2idxt]
        
        cur = 0
        
        for net_index in range(2):
            for node in nets[net_index].nodes():
                for _ in range(self.num_walks):
                    if nets[net_index].degree(node) == 0:
                        continue
                    curr_node = node
                    cur = net_index
                    for _ in range(self.walk_len):
                        if np.random.rand() > self.switch_prob:
                            # choose by degree distribution
                            neighbors = nets[cur].neighbors(curr_node)
                            neighbors_deg = [nets[cur].degree(ele) for ele in neighbors]
                            next_node = np.random.choice(neighbors, p=np.array(neighbors_deg)/np.sum(neighbors_deg))
                            curr_node = next_node
                        else:
                            # switch net now
                            if self.test_dict.get(curr_node) and cur==0:
                                next_node = self.test_dict.get(curr_node)
                                cur = 1 # switched
                                curr_node = next_node
                            elif self.inv_test_dict.get(curr_node) and cur==1:
                                next_node = self.inv_test_dict.get(curr_node)
                                cur = 0 # switched
                                curr_node = next_node
                            else:
                                cur_to_switch = (cur + 1) % 2
                                destination_nodes = nets[cur_to_switch].nodes()
                                if cur == 0:
                                    p = np.array([self.dist_dict[(curr_node, ele)] for ele in destination_nodes])
                                elif cur == 1:
                                    p = np.array([self.dist_dict[(ele, curr_node)] for ele in destination_nodes])
                                p /= p.sum()
                                next_node = np.random.choice(destination_nodes, p=p)
                                cur = cur_to_switch # switched
                                curr_node = next_node
                        if curr_node != node or cur != net_index:
                            pairs.append([id2idices[net_index][node] + suffices[net_index], id2idices[cur][curr_node] + suffices[cur]])
            cur = 1

        print("Done walks")
        return pairs


class LinkPredModel(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(LinkPredModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

    def forward(self, input):
        return torch.sigmoid(self.linear(input))

"""
for d in 1
do
    PD=$HOME/dataspace/graph/email_univ
    PREFIX2=del-nodes-p${d}-seed1
    TRAINRATIO=0.1

    python network_alignment.py \
    --source_dataset ${PD}/graphsage/ \
    --target_dataset ${PD}/${PREFIX2}/graphsage/ \
    --groundtruth ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.test.dict \
    CENALP \
    --train_dict ${PD}/${PREFIX2}/dictionaries/node,split=${TRAINRATIO}.train.dict \
    --num_walks 1 \
    --walk_len 5 \
    --batch_size 512 \
    --threshold 0.5 \
    --linkpred_epochs 0 \
    --num_pair_toadd 10 \
    --num_sample 300 \
    --cuda 
    "909" "16"
done
"""
