import numpy as np
import pickle
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
from torch_geometric.datasets import Reddit2, Flickr, WebKB, WikipediaNetwork, Actor, Amazon, Planetoid
import scipy.sparse as sp
import argparse
from tqdm import tqdm
import time
import math
import sys
import os

import torch
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor, remove_diag, set_diag, spmm, spspmm, get_diag
from torch_sparse import sum as ts_sum
from torch_sparse import mul as ts_mul
from torch_sparse import transpose as ts_trans

from dataset_utils import *
from utils_general import fix_seed, train_test_split, load_npz_dataset
from config import base_dir as data_root

def minibatch_normalization(adj,N,device,k=1000000):
    deg = ts_sum(adj, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.)
    deg_inv_col = deg_inv_sqrt.view(N, 1).to(device)
    deg_inv_row = deg_inv_sqrt.view(1, N).to(device)

    adj = adj.coo()
    for i in tqdm(range(len(adj[0])//k+1)): 
        tmp = SparseTensor(row=adj[0][i*k:(i+1)*k], 
                           col=adj[1][i*k:(i+1)*k], 
                           sparse_sizes=(N,N)).to(device)
        tmp = ts_mul(tmp, deg_inv_col)
        tmp = ts_mul(tmp, deg_inv_row).to("cpu").coo()
        if i == 0:
            adj_t = [tmp[0],tmp[1],tmp[2]]
        else:
            for _ in range(3):
                adj_t[_] = torch.concat([adj_t[_],tmp[_]],dim=0)
    adj_t = SparseTensor(row=adj_t[0],
                         col=adj_t[1],
                         value=adj_t[2],
                         sparse_sizes=(N,N))
    del deg_inv_col, deg_inv_row, tmp
    torch.cuda.empty_cache()
    return adj_t

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def main(args, seed=100):
    fix_seed(seed)
    device = torch.device(f"cuda:{args.cuda}") if args.cuda >= 0 else torch.device("cpu")

    #Load dataset
    if args.dataset[:4] == "ogbn":
        dataset = PygNodePropPredDataset(args.dataset,root=data_root+"dataset/")
        data = dataset[0]
        edge_index = data.edge_index
        N = data.num_nodes
        labels = data.y.data
        d = data.num_features
        features = data.x
        #get split indices
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        del data
        
    elif args.dataset in ["fb100","deezer-europe","arxiv-year","pokec","snap-patents","yelp-chi","genius","twitch-gamer","wiki"]:
        if args.dataset == "fb100":
            sub_dataname = "Penn94"
            dataset = load_fb100_dataset(sub_dataname)
        elif args.dataset == "deezer-europe":
            dataset = load_deezer_dataset()
        elif args.dataset == "arxiv-year":
            dataset = load_arxiv_year_dataset()
        elif args.dataset == "pokec":
            dataset = load_pokec_mat()
        elif args.dataset == "snap-patents":
            dataset = load_snap_patents_mat()
        elif args.dataset == "yelp-chi":
            dataset = load_yelpchi_dataset()
        elif args.dataset == "genius":
            dataset = load_genius()
        elif args.dataset == "twitch-gamer":
            dataset = load_twitch_gamer_dataset() 
        elif args.dataset == "wiki":
            dataset = load_wiki()
        edge_index = dataset.graph["edge_index"]
        N = dataset.graph["num_nodes"]
        labels = dataset.label
        features = dataset.graph["node_feat"]
        d = dataset.graph["node_feat"].shape[1]
        #get split indices
        split_idx = dataset.get_idx_split(train_prop=.5, valid_prop=.25, seed=seed)
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # print(train_idx)
        del dataset
    elif args.dataset in ["squirrel-filtered-directed","chameleon-filtered-directed"]:
        data_dgl = Dataset(name=args.dataset)
        edge_index =  torch.stack(list(data_dgl.graph.edges()),dim=0).long()
        N = data_dgl.graph.num_nodes()
        features = data_dgl.node_features
        labels = data_dgl.labels
        d = features.shape[1]
        split_idx = dict()
        train_idx = data_dgl.train_idx_list[int(seed/100)]
        valid_idx = data_dgl.val_idx_list[int(seed/100)]
        test_idx = data_dgl.test_idx_list[int(seed/100)]
        del data_dgl
    elif args.dataset in ["cora_ml","citeseer"]:
        graph = load_npz_dataset(data_root+"dataset/"+args.dataset+"/"+args.dataset)
        edge_index = graph['edge_index']
        labels = graph['labels']
        N = len(labels)
        features = graph['features']
        d = features.shape[1]
        num_classes = len(set(labels.numpy()))
        mask = train_test_split(labels.numpy(), seed, train_examples_per_class=20, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=500, test_size=N-500-20*num_classes)
        train_idx = torch.flatten((mask['train']==True).nonzero())
        valid_idx = torch.flatten((mask['valid']==True).nonzero())
        test_idx = torch.flatten((mask['test']==True).nonzero())
        del graph
    else:
        split_flag = False
        if args.dataset == "reddit":
            dataset = Reddit2(root=data_root+"dataset/"+args.dataset+"/")
        elif args.dataset == "flickr":
            dataset = Flickr(root=data_root+"dataset/"+args.dataset+"/")
        elif args.dataset == "actor":
            dataset = Actor(root=data_root+"dataset/"+args.dataset+"/")
        else:
            if args.dataset in ["wisconsin", "cornell", "texas"]:
                data_func = WebKB
                split_flag = True # there are several default splits
            elif args.dataset in ["chameleon","squirrel"]:
                data_func = WikipediaNetwork
                split_flag = True # there are several default splits
            elif args.dataset in ["computers","photo"]:
                data_func = Amazon
            dataset = data_func(data_root+"dataset/"+args.dataset+"/", args.dataset)
            
        data = dataset[0]
        if split_flag == True:
            train_idx = torch.flatten((torch.t(data.train_mask)[0]==True).nonzero())
            valid_idx = torch.flatten((torch.t(data.val_mask)[0]==True).nonzero())
            test_idx = torch.flatten((torch.t(data.test_mask)[0]==True).nonzero())
        else:
            train_idx = torch.flatten((data.train_mask==True).nonzero())
            valid_idx = torch.flatten((data.val_mask==True).nonzero())
            test_idx = torch.flatten((data.test_mask==True).nonzero())
            
        edge_index = data.edge_index
        N = data.num_nodes
        labels = data.y.data
        features = data.x
        d = data.num_features
        del data
            
    t_start = time.time()
    num_edge = len(edge_index[0])
    row,col = edge_index
    adj = SparseTensor(row=row,col=col,sparse_sizes=(N,N))
    adj = adj.to_scipy(layout="csr")
    del edge_index

    if not os.path.exists(data_root+"precomputation_data/"+args.dataset):
        os.mkdir(data_root+"precomputation_data/"+args.dataset)
    feat = features.numpy()
    feat = torch.from_numpy(feat).float()
    with open(data_root+"precomputation_data/"+args.dataset+"/feature_training_"+str(seed)+".pickle","wb") as fopen:
        pickle.dump(feat[train_idx,:],fopen)
    with open(data_root+"precomputation_data/"+args.dataset+"/feature_validation_"+str(seed)+".pickle","wb") as fopen:
        pickle.dump(feat[valid_idx,:],fopen)
    with open(data_root+"precomputation_data/"+args.dataset+"/feature_test_"+str(seed)+".pickle","wb") as fopen:
        pickle.dump(feat[test_idx,:],fopen)
    del feat

    if args.dataset == "ogbn-papers100M":
        a = 3
        b = 10
        sep_att = 16
    elif args.dataset == "wiki":
        a = 1
        b = 2
        sep_att = 1  
    else:
        a=b=sep_att = 1

    torch.tensor([0]).to(device)
    filt = args.filter
    if filt == "adjacency":
        adj = adj + adj.transpose()
        adj = adj.tocoo()
        adj = SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), sparse_sizes=(N,N))

        adj_tag = "adj"
        t_tmp = time.time()
        adj = set_diag(adj,1)
        k=adj.nnz()//a+1
        adj_mat = minibatch_normalization(adj,N,device,k=k)
    elif filt == "exact1hop":
        adj = adj + adj.transpose()
        adj = adj.tocoo()
        adj = SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), sparse_sizes=(N,N))

        adj_tag = "adj_i"
        t_tmp = time.time()
        adj = remove_diag(adj,0)
        k=adj.nnz()//a+1
        adj_mat = minibatch_normalization(adj,N,device,k=k)
    elif filt == "adjacency_di":
        adj = adj.tocoo()
        adj = SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), value=torch.ones(len(adj.row)), sparse_sizes=(N,N))
    
        adj_tag = "adj_di"
        t_tmp = time.time()
        # adj_mat = minibatch_normalization(remove_diag(adj,0),N,device,k=k)
        k = adj.nnz()//a+1
        adj_mat = adj
    elif filt == "exact1hop_di":
        adj = adj.tocoo()
        adj = SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), value=torch.ones(len(adj.row)), sparse_sizes=(N,N))
        
        adj_tag = "adj_di_i"
        t_tmp = time.time()
        # adj_mat = minibatch_normalization(remove_diag(adj,0),N,device,k=k)
        adj = remove_diag(adj,0)
        k = adj.nnz()//a+1
        adj_mat = adj
    elif filt == "adjacency_di_t":
        adj = adj.transpose().tocoo()
        adj = SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), value=torch.ones(len(adj.row)), sparse_sizes=(N,N))
    
        adj_tag = "adj_di_t"
        t_tmp = time.time()
        k = adj.nnz()//a+1
        adj_mat = adj
    elif filt == "exact1hop_di_t":
        adj = adj.transpose().tocoo()
        adj = SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), value=torch.ones(len(adj.row)), sparse_sizes=(N,N))
        
        adj_tag = "adj_di_i_t"
        t_tmp = time.time()
        adj = remove_diag(adj,0)
        k = adj.nnz()//a+1
        adj_mat = adj
    
    del adj

    agg_feat = features.numpy()
    agg_feat = torch.from_numpy(agg_feat).float()

    adj_mat = adj_mat.coo()
    #  / 1e9))
    tmp = SparseTensor(row=adj_mat[0][0:k],
                                  col=adj_mat[1][0:k],
                                  value=adj_mat[2][0:k],
                                  sparse_sizes=(N,N))
    tmp = tmp.coo() 
    k_adj = math.ceil(len(adj_mat[0])//b)
    k_att = math.ceil(d//sep_att)

    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(args.layer): # number of hops
            t_tmp = time.time()
            feat_list=[]
            for i_feat in tqdm(range(sep_att)):
                agg_feat_block = agg_feat[:,i_feat*(k_att):(i_feat+1)*(k_att)]
                agg_feat_block = agg_feat_block.to(device)
                for i_adj in range(b): # matrix separation for saving GPU memory
                    tmp = torch.sparse.FloatTensor(
                        torch.stack([adj_mat[0][i_adj*k_adj:(i_adj+1)*k_adj],adj_mat[1][i_adj*k_adj:(i_adj+1)*k_adj]]),
                        adj_mat[2][i_adj*k_adj:(i_adj+1)*k_adj],
                        [N,N]).to(device)
                    torch.cuda.empty_cache()
                    if i_adj == 0:
                        tmp_feat = torch.spmm(tmp, agg_feat_block)
                    else:
                        tmp_feat += torch.spmm(tmp, agg_feat_block)
                feat_list.append(tmp_feat.to("cpu"))
                del tmp_feat, agg_feat_block
                torch.cuda.empty_cache()
            agg_feat = torch.concat(feat_list, dim=1)
            print(str(_+1)+"hop finished")

            with open(data_root+"precomputation_data/"+args.dataset+"/"+filt+"_"+str(_+1)+"_training_"+str(seed)+".pickle","wb") as fopen:
                pickle.dump(agg_feat[train_idx,:],fopen)
            with open(data_root+"precomputation_data/"+args.dataset+"/"+filt+"_"+str(_+1)+"_validation_"+str(seed)+".pickle","wb") as fopen:
                pickle.dump(agg_feat[valid_idx,:],fopen)
            with open(data_root+"precomputation_data/"+args.dataset+"/"+filt+"_"+str(_+1)+"_test_"+str(seed)+".pickle","wb") as fopen:
                pickle.dump(agg_feat[test_idx,:],fopen)

    del agg_feat, adj_mat, tmp
    print(f"GPU max memory usage: {torch.cuda.max_memory_allocated(device=device)/10**9}")

    #save labels
    split_idx = dict()
    split_idx["train"] = train_idx
    split_idx["valid"] = valid_idx
    split_idx["test"] = test_idx
    with open(data_root+"precomputation_data/"+args.dataset+"/labels_"+str(seed)+".pickle","wb") as fopen:
        pickle.dump([split_idx, labels.reshape(-1).long()], fopen)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="ogbn-arxiv",help="name of dataset you use")
    parser.add_argument("--filter",type=str,
                        choices=["adjacency","exact1hop"],
                        default="adjacency",
                        help="graph filter for convolution")
    parser.add_argument("--layer",type=int,default=5,help="number of hops")
    parser.add_argument("--cuda", default=1, type=int, help="which GPU to use")

    args = parser.parse_args()
    main(args)
