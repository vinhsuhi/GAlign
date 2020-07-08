from algorithms.PALE.loss import MappingLossFunctions

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


