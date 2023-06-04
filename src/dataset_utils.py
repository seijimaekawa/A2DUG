#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import math
import pickle
import pandas as pd
import os

import os.path as osp
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from os import path
import gdown
from data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, dataset_drive_url, load_twitch, load_fb100, load_twitch_gamer
import scipy.io
from sklearn.preprocessing import label_binarize

# from cSBM_dataset import dataset_ContextualSBM
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Reddit2
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import Amazon, WikipediaNetwork, Actor, Yelp #, AttributedGraphDataset
from torch_geometric.nn import APPNP
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import is_undirected, to_undirected
from torch_geometric.io import read_npz
# from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from ogb.nodeproppred import NodePropPredDataset

# data_root = '/home/maekawa/simpleGCN/FSGNN-main/large_data/dataset/'
# data_root = '../dataset/'
data_root = '/home/maekawa/class_imbalance/dataset/'

import dgl
from dgl import ops
from sklearn.metrics import roc_auc_score


class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu', use_sgc_features=False, use_identity_features=False,
                 use_adjacency_features=False, do_not_use_original_features=False):

        if do_not_use_original_features and not any([use_sgc_features, use_identity_features, use_adjacency_features]):
            raise ValueError('If original node features are not used, at least one of the arguments '
                             'use_sgc_features, use_identity_features, use_adjacency_features should be used.')

        print('Preparing data...')
        data = np.load(os.path.join(data_root, f'{name.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)

        if 'directed' not in name:
            graph = dgl.to_bidirected(graph)

        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        node_features = self.augment_node_features(graph=graph,
                                                   node_features=node_features,
                                                   use_sgc_features=use_sgc_features,
                                                   use_identity_features=use_identity_features,
                                                   use_adjacency_features=use_adjacency_features,
                                                   do_not_use_original_features=do_not_use_original_features)

        self.name = name
        self.device = device

        self.graph = graph.to(device)
        self.node_features = node_features.to(device)
        self.labels = labels.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1]
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits

    def compute_metrics(self, logits):
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    @staticmethod
    def augment_node_features(graph, node_features, use_sgc_features, use_identity_features, use_adjacency_features,
                              do_not_use_original_features):

        n = graph.num_nodes()
        original_node_features = node_features

        if do_not_use_original_features:
            node_features = torch.tensor([[] for _ in range(n)])

        if use_sgc_features:
            sgc_features = Dataset.compute_sgc_features(graph, original_node_features)
            node_features = torch.cat([node_features, sgc_features], axis=1)

        if use_identity_features:
            node_features = torch.cat([node_features, torch.eye(n)], axis=1)

        if use_adjacency_features:
            graph_without_self_loops = dgl.remove_self_loop(graph)
            adj_matrix = graph_without_self_loops.adjacency_matrix().to_dense()
            node_features = torch.cat([node_features, adj_matrix], axis=1)

        return node_features

    @staticmethod
    def compute_sgc_features(graph, node_features, num_props=5):
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        degrees = graph.out_degrees().float()
        degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
        norm_coefs = 1 / degree_edge_products ** 0.5

        for _ in range(num_props):
            node_features = ops.u_mul_e_sum(graph, node_features, norm_coefs)

        return node_features


class dataset_heterophily(InMemoryDataset):
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        if str(type(self.data.train_percent)) == "<class 'torch.Tensor'>":
            self.train_percent = self.data.train_percent.item()
        else:
            self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class WebKB(InMemoryDataset):
    r"""The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def DataLoader(name):
    # assert name in ['cSBM_data_Aug_19_2020-13:06',
    #                 'cSBM_data_Aug_18_2020-18:50',
    #                 'cSBM_data_Aug_21_2020-10:06',
    #                 'cSBM_data_Aug_19_2020-20:41',
    #                 'cSBM_data_Aug_21_2020-11:04',
    #                 'cSBM_data_Aug_21_2020-11:21',
    #                 'cSBM_data_Sep_01_2020-14:15',
    #                 'cSBM_data_Sep_01_2020-14:18',
    #                 'cSBM_data_Sep_01_2020-14:19',
    #                 'cSBM_data_Sep_01_2020-14:32',
    #                 'cSBM_data_Sep_01_2020-14:22',
    #                 'cSBM_data_Sep_01_2020-14:23',
    #                 'cSBM_data_Sep_01_2020-14:27',
    #                 'cSBM_data_Sep_01_2020-14:29',
    #                 'Cora', 'Citeseer', 'PubMed',
    #                 'Computers', 'Photo',
    #                 'chameleon', 'film', 'squirrel',
    #                 'Texas', 'Cornell']

    # if name in ['cSBM_data_Aug_19_2020-13:06',
    #             'cSBM_data_Aug_18_2020-18:50',
    #             'cSBM_data_Aug_21_2020-10:06',
    #             'cSBM_data_Aug_19_2020-20:41',
    #             'cSBM_data_Aug_21_2020-11:04',
    #             'cSBM_data_Aug_21_2020-11:21',
    #             'cSBM_data_Sep_01_2020-14:15',
    #             'cSBM_data_Sep_01_2020-14:18',
    #             'cSBM_data_Sep_01_2020-14:19',
    #             'cSBM_data_Sep_01_2020-14:32',
    #             'cSBM_data_Sep_01_2020-14:22',
    #             'cSBM_data_Sep_01_2020-14:23',
    #             'cSBM_data_Sep_01_2020-14:27',
    #             'cSBM_data_Sep_01_2020-14:29']:
    # if 'cSBM_data' in name:
    #     path = './data/'
    #     dataset = dataset_ContextualSBM(path, name=name)
    # else:
    name = name.lower()

    if name in ['cora', 'citeseer', 'pubmed']:
        # root_path = './'
        # path = osp.join(root_path, 'data', name)
        dataset = Planetoid(data_root+name+"/", name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        # root_path = './'
        # path = osp.join(root_path, 'data', name)
        dataset = Amazon(data_root+name+"/", name, T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=data_root+name+"/",
                        name=name, geom_gcn_preprocess=True
                                   , transform=T.NormalizeFeatures())
        # dataset = dataset_heterophily(
        #     root='./data/', name=name, transform=T.NormalizeFeatures())
    elif name in ['actor']:
        # root_path = './'
        dataset = Actor(root=data_root+name+"/",
                        transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root=data_root+name+"/",
                        name=name, transform=T.NormalizeFeatures())
        
    elif name in ['ogbn-arxiv']:
        # dataset = PygNodePropPredDataset(name, root='./data/')
        dataset = PygNodePropPredDataset(name, root=data_root, transform=T.NormalizeFeatures())

        # dataset[0].x = preprocess_features(dataset[0].x)
    elif name in ['blogcatalog', 'wiki']:
        dataset = AttributedGraphDataset(
            root=data_root+name+"/", name=name, transform=T.NormalizeFeatures())
    elif name == "flickr":
        dataset = Flickr(root=data_root+name+"/", transform=T.NormalizeFeatures())
    elif name == "yelp":
        dataset = Yelp(root=data_root+name+"/", transform=T.NormalizeFeatures())
    elif name == "reddit":
        dataset = Reddit2(root=data_root+name+"/", transform=T.NormalizeFeatures())
        
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset


class NCDataset(object):
    def __init__(self, name, root=f'{data_root}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None
    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, seed=100):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """
        def fix_seed(seed):
            import random
            import numpy as np
            import torch
            # random
            random.seed(seed)
            # Numpy
            np.random.seed(seed)
            # Pytorch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
        fix_seed(seed)
        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
        
def load_nc_dataset(dataname, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'twitch-e':
        # twitch-explicit graph
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(sub_dataname)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset()
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset()
    elif dataname == 'arxiv-year':
        dataset = load_arxiv_year_dataset()
    elif dataname == 'pokec':
        dataset = load_pokec_mat()
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat()
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset()
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(dataname)
    elif dataname in ('Cora', 'CiteSeer', 'PubMed'):
        dataset = load_planetoid_dataset(dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(dataname)
    elif dataname == "genius":
        dataset = load_genius()
    elif dataname == "twitch-gamer":
        dataset = load_twitch_gamer_dataset() 
    elif dataname == "wiki":
        dataset = load_wiki()
    else:
        raise ValueError('Invalid dataname')
    return dataset

def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{data_root}deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(nclass=5):
    filename = 'arxiv-year'
    dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(root=data_root, name='ogbn-arxiv')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    return dataset


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not path.exists(f'{data_root}pokec.mat'):
        gdown.download(id=dataset_drive_url['pokec'], \
            output=f'{data_root}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{data_root}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset

def load_snap_patents_mat(nclass=5):
    if not path.exists(f'{data_root}snap_patents.mat'):
        p = dataset_drive_url['snap-patents']
        print(f"Snap patents url: {p}")
        gdown.download(id=dataset_drive_url['snap-patents'], \
            output=f'{data_root}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{data_root}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_yelpchi_dataset():
    if not path.exists(f'{data_root}YelpChi.mat'):
        gdown.download(id=dataset_drive_url['yelp-chi'], \
            output=f'{data_root}YelpChi.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{data_root}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset

def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'{data_root}genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_twitch_gamer_dataset(task="mature", normalize=True):
    if not path.exists(f'{data_root}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_feat'],
            output=f'{data_root}twitch-gamer_feat.csv', quiet=False)
    if not path.exists(f'{data_root}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url['twitch-gamer_edges'],
            output=f'{data_root}twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{data_root}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{data_root}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_wiki():

    if not path.exists(f'{data_root}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_features'], \
            output=f'{data_root}wiki_features2M.pt', quiet=False)
    
    if not path.exists(f'{data_root}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_edges'], \
            output=f'{data_root}wiki_edges2M.pt', quiet=False)

    if not path.exists(f'{data_root}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url['wiki_views'], \
            output=f'{data_root}wiki_views2M.pt', quiet=False)


    dataset = NCDataset("wiki") 
    features = torch.load(f'{data_root}wiki_features2M.pt')
    edges = torch.load(f'{data_root}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{data_root}wiki_views2M.pt') 
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges, 
                     "edge_feat": None, 
                     "node_feat": features, 
                     "num_nodes": num_nodes}
    dataset.label = label 
    return dataset 