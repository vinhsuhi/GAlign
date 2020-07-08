# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
from torch import optim
import re
import inspect
import pdb
import numpy as np
import torch.nn.functional as F


logger = getLogger()


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.numpy()




def get_candidates(emb1, emb2, p_keep):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    
    # contextual dissimilarity measure
    if True:
        knn = '10'
        assert knn.isdigit()
        knn = int(knn)
        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)


        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)
            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
        ], 1)
    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)
    
    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]
    
    len_all_pairs = len(all_pairs)
    num_keeps = int(p_keep * len_all_pairs)
    all_scores = all_scores[:num_keeps]
    all_pairs = all_pairs[:num_keeps]
    return all_pairs


def build_dictionary(src_emb, tgt_emb, s2t_candidates=None, t2s_candidates=None, p_keep=1):
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = True
    t2s = False
    assert s2t or t2s

    if s2t:
        if s2t_candidates is None:
            s2t_candidates = get_candidates(src_emb, tgt_emb, p_keep)
    if t2s:
        if t2s_candidates is None:
            t2s_candidates = get_candidates(tgt_emb, src_emb, p_keep)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    # if params.dico_build == 'S2T':
    dico = s2t_candidates
    # logger.info('New train dictionary of %i pairs.' % dico.size(0))
    return dico.cuda()



def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params



def read_embedding_from_file(id2idx, path, embedding_dim, cuda):
    embedding = np.zeros((len(id2idx), embedding_dim))
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                node, vect = line.rstrip().split(" ", 1)
                vect = np.fromstring(vect, sep=' ')
                embedding[id2idx[node]] = vect
    embedding = torch.FloatTensor(embedding)
    if cuda:
        embedding = embedding.cuda()
    embedding = F.normalize(embedding, dim = 1)
    return embedding


def to_word2vec_format(val_embeddings, nodes, out_dir, filename, dim, id2idx, pref=""):
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



# def check_edge_in_edges(edge, edges):
#     for e in edges:
#         if np.array_equal(edge, e):
#             return True
#     return False


# def extend_edge(self, source_edges, target_edges):
#     for edge in source_edges:
#         if edge[0] in self.pale_train_anchors.keys():
#             if edge[1] in self.pale_train_anchors.keys():
#                 if not self.check_edge_in_edges(np.array([self.pale_train_anchors[edge[0]], self.pale_train_anchors[edge[1]]]), target_edges):
#                     target_edges = np.concatenate((target_edges, np.array(([[self.pale_train_anchors[edge[0]], self.pale_train_anchors[edge[1]]]]))), axis=0)
#                     target_edges = np.concatenate((target_edges, np.array(([[self.pale_train_anchors[edge[1]], self.pale_train_anchors[edge[0]]]]))), axis=0)

#     inverse_gt_train = {v:k for k, v in self.pale_train_anchors.items()}
#     for edge in target_edges:
#         if edge[0] in self.pale_train_anchors.values():
#             if edge[1] in self.pale_train_anchors.values():
#                 if not self.check_edge_in_edges(np.array([inverse_gt_train[edge[0]], inverse_gt_train[edge[1]]]), source_edges):
#                     source_edges = np.concatenate((source_edges, np.array(([[inverse_gt_train[edge[0]], inverse_gt_train[edge[1]]]]))), axis=0)
#                     source_edges = np.concatenate((source_edges, np.array(([[inverse_gt_train[edge[1]], inverse_gt_train[edge[0]]]]))), axis=0)
#     return source_edges, target_edges