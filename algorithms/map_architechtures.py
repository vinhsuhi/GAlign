from algorithms.PALE.loss import MappingLossFunctions
from torch.nn import init

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def autoencoder_loss(decoded, source_feats, inversed_decoded, target_feats):
    num_examples1 = source_feats.shape[0]
    num_examples2 = target_feats.shape[0]
    straight_loss = (num_examples1 - (decoded * source_feats).sum())/num_examples1
    inversed_loss = (num_examples2 - (inversed_decoded * target_feats).sum())/num_examples2
    loss = straight_loss + inversed_loss
    return loss


def init_weight(modules):
    activation = 'relu'
    for m in modules:
        print(m)
        if isinstance(m, nn.Linear):
            m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


##############################################################################
#               MAPPING MODELS
##############################################################################

class PaleMapping(nn.Module):
    def __init__(self, source_embedding, target_embedding):
        """
        Parameters
        ----------
        source_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for nodes
        target_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for target_nodes
        target_neighbor: dict
            dict of target_node -> target_nodes_neighbors. Used for calculate vinh_loss
        """

        super(PaleMapping, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.loss_fn = MappingLossFunctions()
    

class PaleMappingLinear(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding):
        super(PaleMappingLinear, self).__init__(source_embedding, target_embedding)
        self.maps = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping = self.forward(source_feats)

        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        
        return mapping_loss

    def forward(self, source_feats):
        ret = self.maps(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret



class PaleMappingMlp(PaleMapping):
    def __init__(self, embedding_dim, source_embedding, target_embedding, activate_function='sigmoid'):

        super(PaleMappingMlp, self).__init__(source_embedding, target_embedding)

        if activate_function == 'sigmoid':
            self.activate_function = nn.Sigmoid()
        elif activate_function == 'relu':
            self.activate_function = nn.ReLU()
        else:
            self.activate_function = nn.Tanh()

        hidden_dim = 2*embedding_dim
        self.mlp = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim, bias=True),
            self.activate_function,
            nn.Linear(hidden_dim, embedding_dim, bias=True)
        ])


    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping = self.forward(source_feats)

        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size
        return mapping_loss


    def forward(self, source_feats):
        ret = self.mlp(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret




################
#NAWAL
################


class Discriminator(nn.Module):
    def __init__(self, emb_dim, dis_layers, dis_hid_dim, dis_dropout, dis_input_dropout):
        super(Discriminator, self).__init__()

        self.emb_dim = emb_dim
        self.dis_layers = dis_layers
        self.dis_hid_dim = dis_hid_dim
        self.dis_dropout = dis_dropout
        self.dis_input_dropout = dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


class Mapping(nn.Module):
    def __init__(self, emb_dim):
        super(Mapping, self).__init__()

        self.emb_dim = emb_dim
        self.layer = nn.Linear(emb_dim, emb_dim, bias=False)
        self.layer.weight.data.copy_(torch.diag(torch.ones(emb_dim)))
    
    def forward(self, x):
        return F.normalize(self.layer(x))


###################33
# AAE mapping
########################

class EncoderMLP(nn.Module):
    def __init__(self, emb_dim, N):
        super(EncoderMLP, self).__init__()
        embedding_dim = emb_dim
        self.lin1 = nn.Linear(embedding_dim, N, bias=False)
        self.lin2 = nn.Linear(N, N, bias=False)
        self.lin3 = nn.Linear(N, embedding_dim, bias=False)
        self.params = [self.lin1, self.lin2, self.lin3]

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x_out = self.lin3(x)
        return x_out
        

class DecoderMLP(nn.Module):
    def __init__(self, emb_dim, N):
        super(DecoderMLP, self).__init__()
        input_dim = 300
        embedding_dim = emb_dim
        self.lin1 = nn.Linear(input_dim, N, bias=False)
        self.lin2 = nn.Linear(N, N, bias=False)
        self.lin3 = nn.Linear(N, embedding_dim, bias=False)
        self.params = [self.lin1, self.lin2, self.lin3]

    
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x_out = self.lin3(x)
        return x_out


class DecoderLinear(nn.Module):
    def __init__(self, emb_dim):
        super(DecoderLinear, self).__init__()
        self.emb_dim = emb_dim
        self.layer = nn.Linear(emb_dim, emb_dim, bias=False)
        self.params = [self.layer]

        #self.layer.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
    
    def forward(self, x):
        return self.layer(x)


class EncoderLinear(nn.Module):
    def __init__(self, emb_dim):
        super(EncoderLinear, self).__init__()
        self.emb_dim = emb_dim
        self.layer = nn.Linear(emb_dim, emb_dim, bias=False)
        self.params = [self.layer]
        
        # self.layer.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))
    
    def forward(self, x):
        return self.layer(x)


class MappingModel(nn.Module):
    def __init__(self, embedding_dim=800, hidden_dim1=1200, hidden_dim2=1600, source_embedding=None, target_embedding=None):
        """
        Parameters
        ----------
        embedding_dim: int
            Embedding dim of nodes
        hidden_dim1: int
            Number of hidden neurons in the first hidden layer of MLP
        hidden_dim2: int
            Number of hidden neurons in the second hidden layer of MLP
        source_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for source nodes
        target_embedding: torch.Tensor or Embedding_model
            Used to get embedding vectors for target nodes
        """

        super(MappingModel, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding

        # theta is a MLP nn (encoder)
        self.theta = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])
        # inversed_theta is a MLP nn (decoder)
        self.inversed_theta = nn.Sequential(*[
            nn.Linear(embedding_dim, hidden_dim1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim2, embedding_dim, bias=True)
        ])
        init_weight(self.modules())


    def forward(self, source_feats, mode='t'):
        encoded = self.theta(source_feats)
        encoded = F.normalize(encoded, dim=1)
        if mode != 't':
            return encoded
        decoded = self.inversed_theta(encoded)
        decoded = F.normalize(decoded, dim=1)
        return decoded


    def inversed_forward(self, target_feats):
        inversed_encoded = self.inversed_theta(target_feats)
        inversed_encoded = F.normalize(inversed_encoded, dim=1)
        inversed_decoded = self.theta(inversed_encoded)
        inversed_decoded = F.normalize(inversed_decoded, dim=1)
        return inversed_decoded


    def supervised_loss(self, source_batch, target_batch, alpha=1, k=5):
        source_feats = self.source_embedding[source_batch]
        target_feats = self.target_embedding[target_batch]

        source_after_map = self.theta(source_feats)
        source_after_map = F.normalize(source_after_map)

        target_after_map = self.inversed_theta(target_feats)
        target_after_map = F.normalize(target_after_map, dim=1)

        reward_source_target = 0
        reward_target_source = 0

        for i in range(source_feats.shape[0]):
            embedding_of_ua = source_feats[i]
            embedding_of_target_of_ua = target_feats[i]
            embedding_of_ua_after_map = source_after_map[i]
            reward_source_target += torch.sum(embedding_of_ua_after_map * embedding_of_target_of_ua)
            top_k_simi = self.find_topk_simi(embedding_of_target_of_ua, self.target_embedding, k=k)
            reward_source_target += self.compute_rst(embedding_of_ua_after_map, top_k_simi)
            reward_target_source += self.compute_rts(embedding_of_ua, top_k_simi)
        st_loss = -alpha*reward_source_target/source_feats.shape[0]
        ts_loss = -(1-alpha)*reward_target_source/target_feats.shape[0]
        loss = st_loss + ts_loss
        return loss


    def unsupervised_loss(self, source_batch, target_batch):
        source_feats = self.source_embedding[source_batch]
        target_feats = self.target_embedding[target_batch]
        decoded = self.forward(source_feats)
        inversed_decoded = self.inversed_forward(target_feats)
        loss = autoencoder_loss(decoded, source_feats, inversed_decoded, target_feats)
        return loss


    def compute_rst(self, embedding_of_ua_after_map, top_k_simi):
        top_k_embedding = self.target_embedding[top_k_simi]
        cosin = torch.sum(embedding_of_ua_after_map * top_k_embedding, dim=1)
        reward = torch.mean(torch.log(cosin + 1))
        return reward


    def compute_rts(self, embedding_of_ua, top_k_simi):
        top_k_embedding = self.target_embedding[top_k_simi]
        top_k_simi_after_inversed_map = self.inversed_theta(top_k_embedding)
        top_k_simi_after_inversed_map = F.normalize(top_k_simi_after_inversed_map, dim=1)
        cosin = torch.sum(embedding_of_ua * top_k_simi_after_inversed_map, dim=1)
        reward = torch.mean(torch.log(cosin + 1))
        return reward


    def find_topk_simi(self, embedding_of_ua_after_map, target_embedding, k):
        cosin_simi_matrix = torch.matmul(embedding_of_ua_after_map, target_embedding.t())
        top_k_index = cosin_simi_matrix.sort()[1][-k:]
        return top_k_index




