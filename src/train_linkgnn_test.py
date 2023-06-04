import time
import random
import argparse
import numpy as np
import uuid
import pickle
import sys
from tqdm import tqdm
import json
import os
import scipy.sparse as sp

## torch ##
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_sparse import SparseTensor, matmul, set_diag
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from sklearn.preprocessing import normalize as sk_normalize

## import from this repo ##
from utils_general import *
from model import *
from config import *
from hermitian import hermitian_decomp_sparse, cheb_poly_sparse

## for hyperparameter tune ##
import optuna

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
# linkgnn_flag = 'linkgnn_'
linkgnn_flag = '' # 'simplink'

def main(args):
    def create_batch(input_data):
        num_sample = input_data[0].shape[0]
        list_bat = []
        for i in range(0,num_sample,batch_size):
            if (i+batch_size)<num_sample:
                list_bat.append((i,i+batch_size))
            else:
                list_bat.append((i,num_sample))
        return list_bat


    def test(model,st,end):
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            if args.linkgnn:
                output, out_AUC = model(features[split_idx['test']], A_test, A_di_test, A_di_t_test, test_data, device, st, end)
            elif args.model in ["LINKX", "LINK"]:
                output, out_AUC = model(features[split_idx['test']], A_test, device, st, end)
            # elif args.model in ["SGC", "FSGNN"]:
            #     output, out_AUC = model(test_data, device, st, end)
            elif args.model == "GloGNN":
                out = model(features, A)
                output = out[0][split_idx['test']]
                out_AUC = out[1][split_idx['test']]
            elif args.model in ["GCN", "GPRGNN"]:
                output, out_AUC = model(features, edge_index)
                output, out_AUC = output[split_idx["test"]], out_AUC[split_idx["test"]]
            elif args.model == "DGCN":
                out = model(features, edge_index, edge_in, in_weight, edge_out, out_weight)
                output = out[0][split_idx['test']]
                out_AUC = out[1][split_idx['test']]
            elif args.model == "Magnet":
                output, out_AUC = model(args.X_real, args.X_img)
                output, out_AUC = output[0].T[split_idx["test"]], out_AUC[0].T[split_idx["test"]]
            elif args.model in ["Digraph", "DigraphIB"]:
                output, out_AUC = model(args.features, args.edges, args.edge_weight)
                output, out_AUC = output[split_idx["test"]], out_AUC[split_idx["test"]]
            elif args.model == "ACMGCN":
                out = model(features, adj_low, adj_high)
                output = out[0][split_idx['test']]
                out_AUC = out[1][split_idx['test']]
            else:
                output, out_AUC = model(test_data,device,st,end)
            loss_test = F.nll_loss(output, test_labels[st:end])
            return loss_test.item(), output.max(1)[1].type_as(test_labels[st:end]), out_AUC
        
    def train_step(model,optimizer):
        def train(st,end):
            model.train()
            optimizer.zero_grad()
            if args.linkgnn:
                output, out_AUC = model(features[split_idx['train']], A_train, A_di_train, A_di_t_train, train_data, device, st, end)
            elif args.model in ["LINKX", "LINK"]:
                output, out_AUC = model(features[split_idx['train']], A_train, device, st, end)
            # elif args.model in ["SGC", "FSGNN"]:
            #     output, out_AUC = model(train_data, device, st, end)
            elif args.model == "GloGNN":
                out = model(features, A)
                output = out[0][split_idx['train']]
                out_AUC = out[1][split_idx['train']]
            elif args.model in ["GCN", "GPRGNN"]:
                output, out_AUC = model(features, edge_index)
                output, out_AUC = output[split_idx['train']], out_AUC[split_idx['train']]
            elif args.model == "DGCN":
                out = model(features, edge_index, edge_in, in_weight, edge_out, out_weight)
                output = out[0][split_idx['train']]
                out_AUC = out[1][split_idx['train']]
            elif args.model == "Magnet":
                output, out_AUC = model(args.X_real, args.X_img)
                output, out_AUC = output[0].T[split_idx['train']], out_AUC[0].T[split_idx['train']]
            elif args.model in ["Digraph", "DigraphIB"]:
                output, out_AUC = model(args.features, args.edges, args.edge_weight)
                output, out_AUC = output[split_idx['train']], out_AUC[split_idx['train']]
            elif args.model == "ACMGCN":
                out = model(features, adj_low, adj_high)
                output = out[0][split_idx['train']]
                out_AUC = out[1][split_idx['train']]
            else:
                output, out_AUC = model(train_data,device,st,end)
            loss_train = F.nll_loss(output, train_labels[st:end])
            loss_train.backward()
            optimizer.step()
            # print(output)
            return loss_train.item(), output.max(1)[1].type_as(train_labels[st:end]), out_AUC
        def validate(st,end):
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                if args.linkgnn:
                    output, out_AUC = model(features[split_idx['valid']], A_valid, A_di_valid, A_di_t_valid, valid_data, device, st, end)
                elif args.model in ["LINKX", "LINK"]:
                    output, out_AUC = model(features[split_idx['valid']], A_valid, device, st, end)
                # elif args.model in ["SGC", "FSGNN"]:
                #     output, out_AUC = model(valid_data, device, st, end)
                elif args.model == "GloGNN":
                    out = model(features, A)
                    output = out[0][split_idx["valid"]]
                    out_AUC = out[1][split_idx["valid"]]
                elif args.model in ["GCN", "GPRGNN"]:
                    output, out_AUC = model(features, edge_index)
                    output, out_AUC = output[split_idx["valid"]], out_AUC[split_idx["valid"]]
                elif args.model == "DGCN":
                    out = model(features, edge_index, edge_in, in_weight, edge_out, out_weight)
                    output = out[0][split_idx["valid"]]
                    out_AUC = out[1][split_idx["valid"]]
                elif args.model == "Magnet":
                    output, out_AUC = model(args.X_real, args.X_img)
                    output, out_AUC = output[0].T[split_idx["valid"]], out_AUC[0].T[split_idx["valid"]]
                elif args.model in ["Digraph", "DigraphIB"]:
                    output, out_AUC = model(args.features, args.edges, args.edge_weight)
                    output, out_AUC = output[split_idx["valid"]], out_AUC[split_idx["valid"]]
                elif args.model == "ACMGCN":
                    out = model(features, adj_low, adj_high)
                    output = out[0][split_idx["valid"]]
                    out_AUC = out[1][split_idx["valid"]]
                else:
                    output, out_AUC = model(valid_data,device,st,end)
                loss_val = F.nll_loss(output, valid_labels[st:end])
            return loss_val.item(), output.max(1)[1].type_as(valid_labels[st:end]), out_AUC

        bad_counter = 0
        # best = 999999999
        best = 0
        best_epoch = 0
        acc = 0
        # valid_num = valid_data[0][0].shape[0]
        valid_num = valid_labels.shape[0]
        best_dic = dict()

        # for epoch in tqdm(range(args.epochs)):
        for epoch in range(args.epochs):
            list_loss = []
            random.shuffle(list_bat_train)
            pred_train=torch.tensor([])
            out_AUC_train=[]
            label_train=torch.tensor([])
            for st,end in list_bat_train:
                loss_tra, pred, out_AUC = train(st,end)
                pred_train = torch.concat([pred_train, pred.detach().to('cpu')], axis=0)
                if len(out_AUC_train) == 0:
                    out_AUC_train = out_AUC.detach().to('cpu')
                else:
                    out_AUC_train = torch.concat([out_AUC_train,out_AUC.detach().to('cpu')],axis=0)
                label_train = torch.concat([label_train, train_labels[st:end].to('cpu')], axis=0)        
                list_loss.append(loss_tra)
            loss_tra = np.mean(list_loss)
            result_tra = calc_metric(pred_train.numpy(),label_train.numpy(),out_AUC_train.numpy())

            list_loss_val = []
            pred_val = torch.tensor([])
            out_AUC_val = []
            label_val = torch.tensor([])
            for st,end in list_bat_val:
                loss_val, pred, out_AUC = validate(st,end)
                pred_val = torch.concat([pred_val,pred.detach().to('cpu')],axis=0)
                if len(out_AUC_val) == 0:
                    out_AUC_val = out_AUC.detach().to('cpu')
                else:
                    out_AUC_val = torch.concat([out_AUC_val,out_AUC.detach().to('cpu')],axis=0)
                label_val = torch.concat([label_val,valid_labels[st:end].to('cpu')],axis=0)
                list_loss_val.append(loss_val)
            loss_val = np.mean(list_loss_val)
            result_val = calc_metric(pred_val.numpy(),label_val.numpy(),out_AUC_val.numpy())

            #Uncomment to see losses
            if(epoch+1)%50 == 0:
                list_loss_test = []
                pred_test = torch.tensor([])
                out_AUC_test = []
                label_test = torch.tensor([])
                for st,end in list_bat_test:
                    with torch.no_grad():
                        loss_test, pred_pred, out_AUC_test = test(model,st, end)
                    list_loss_test.append(loss_test)
                    pred_test = torch.concat([pred_test, out_AUC_test.detach().to('cpu')], axis=0)
                    label_test = torch.concat([label_test, test_labels[st:end].to('cpu')], axis=0)
                loss_test = np.mean(list_loss_test)
                result_test = calc_metric(pred_test.max(1)[1].type_as(label_test).numpy(),label_test.numpy(),F.softmax(pred_test,dim=1).numpy())
                print('Epoch:{:04d}'.format(epoch+1),
                    'train',
                    'loss:{:.3f}'.format(loss_tra),
                    'acc:{:.2f}'.format(result_tra['accuracy']*100),
                    'f1 macro:{:.2f}'.format(result_tra['macro avg']['f1-score']*100),
                    '| val',
                    'loss:{:.3f}'.format(loss_val),
                    'acc:{:.2f}'.format(result_val['accuracy']*100),
                    'f1 macro:{:.2f}'.format(result_val['macro avg']['f1-score']*100),
                    '| test',
                    'loss:{:.3f}'.format(loss_test),
                    'acc:{:.2f}'.format(result_test['accuracy']*100),
                    'f1 macro:{:.2f}'.format(result_test['macro avg']['f1-score']*100)
                     )
            # if loss_val < best:
                # best = loss_val
            if result_val['accuracy'] > best:
                best = result_val['accuracy']
                best_dic['tra_loss'] = loss_tra
                best_dic['val_loss'] = loss_val
                best_dic['best_epoch'] = epoch
                best_dic['val_result'] = result_val
                best_dic['tra_result'] = result_tra
                best_dic['best_model'] = model
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
        best_dic["epoch"] = epoch
        return best_dic
    
    ### MAIN function ###
    cudaid = "cuda:"+str(args.cuda)
    device = torch.device(cudaid)
    print(device)
    
    ablation_tag = ""
    if args.linkgnn:
        ablation_tag += "_"+args.agg
    if args.wo_att:
        ablation_tag += "_wo_att"
    if args.wo_directed:
        ablation_tag += "_wo_directed"
    if args.wo_undirected:
        ablation_tag += "_wo_undirected"
    if args.wo_adj:
        ablation_tag += "_wo_adj"
    if args.wo_agg:
        ablation_tag += "_wo_agg"
    if args.wo_transpose:
        ablation_tag += "_wo_transpose"
    if args.wo_mlp:
        ablation_tag += "_wo_mlp"
    if args.model in ["GloGNN", "LINK", "LINKX"]:
        if args.directed:
            ablation_tag += "_directed"
        else:
            ablation_tag += "_undirected"
    print(ablation_tag)
    
    ### Precomputation for Feature Aggregation ###
    # data_path = base_dir+'extreme_few_label/precomputation_data/'+args.dataset
    precomp_flag = False    
    
    t_start = time.time()
    
    ### Data Load ###    
    # Load node features used for input model #
    # input graph for linkgnn
    if args.linkgnn:
        train_data, valid_data, test_data, split_idx, label = load_precomp(args, precomp_flag, base_dir+'linkgnn/precomputation_data/'+args.dataset, seed=args.seed)
        args.num_nodes = len(label)
        args.num_features = train_data[0].shape[1]
        edge_index, features, split_idx, label = load_graph(args, base_dir, seed=args.seed)
        
        row, col = edge_index
        row = row-row.min()
        A_di = SparseTensor(row=row, col=col,
                         sparse_sizes=(args.num_nodes, args.num_nodes)
                        )
        
        adj = SparseTensor(row=row,col=col,sparse_sizes=(args.num_nodes, args.num_nodes))
        adj = adj.to_scipy(layout='csr').transpose()
        adj = adj.tocoo()
        row_t = torch.LongTensor(adj.row)
        col_t = torch.LongTensor(adj.col)
        row_t = row_t-row_t.min()
        A_di_t = SparseTensor(row=row_t, col=col_t,
                           sparse_sizes=(args.num_nodes, args.num_nodes)
                          )
        
        edge_index_undi = to_undirected(edge_index)
        row, col = edge_index_undi
        adj = SparseTensor(row=row,col=col,sparse_sizes=(args.num_nodes, args.num_nodes))
        adj = adj.to_scipy(layout='csr')
        adj = adj.tocoo()
        # args.num_nodes = features.shape[0]
        row = row-row.min()
        A = SparseTensor(row=row, col=col,
                         sparse_sizes=(args.num_nodes, args.num_nodes)
                        )
        ## for undirected graphs
        
        ## undirected
        A_train = A[split_idx['train']] #.to_torch_sparse_coo_tensor()
        A_valid = A[split_idx['valid']] #.to_torch_sparse_coo_tensor()
        A_test = A[split_idx['test']] #.to_torch_sparse_coo_tensor()
        
        ## directed
        A_di_train = A_di[split_idx['train']] #.to_torch_sparse_coo_tensor()
        A_di_valid = A_di[split_idx['valid']] #.to_torch_sparse_coo_tensor()
        A_di_test = A_di[split_idx['test']] #.to_torch_sparse_coo_tensor()
        
        ## transposed directed
        A_di_t_train = A_di_t[split_idx['train']] #.to_torch_sparse_coo_tensor()
        A_di_t_valid = A_di_t[split_idx['valid']] #.to_torch_sparse_coo_tensor()
        A_di_t_test = A_di_t[split_idx['test']] #.to_torch_sparse_coo_tensor()
        
    elif args.model in ["GCN", "GPRGNN"]:
        edge_index, features, split_idx, label = load_graph(args, base_dir, seed=args.seed)
        args.num_nodes = features.shape[0]
        args.num_features = features.shape[1]
        if args.model == "GCN":
            row,col = edge_index
            adj = SparseTensor(row=row,col=col,sparse_sizes=(args.num_nodes,args.num_nodes))
            adj = adj.to_scipy(layout='csr')
            print("Getting undirected matrix...")
            adj = adj + adj.transpose()
            print("Saving unnormalized adjacency matrix")
            adj = adj.tocoo()
            adj = normalization(set_diag(SparseTensor(row=torch.LongTensor(adj.row), col=torch.LongTensor(adj.col), sparse_sizes=(args.num_nodes,args.num_nodes)),1)).coo()
            edge_index = torch.sparse.FloatTensor(torch.stack([adj[0],adj[1]]),adj[2], [args.num_nodes,args.num_nodes]).to(device)
        else:
            edge_index = to_undirected(edge_index)
            edge_index = edge_index.to(device)
        features = features.to(device)
    elif args.model == "MLP":
        edge_index, features, split_idx, label = load_graph(args, base_dir, seed=args.seed)
        args.num_nodes = features.shape[0]
        args.num_features = features.shape[1]
        train_data = [features[split_idx['train']]]
        valid_data = [features[split_idx['valid']]]
        test_data = [features[split_idx['test']]]
    elif args.model in ["SGC", "FSGNN"]:
        train_data, valid_data, test_data, split_idx, label = load_precomp(args, precomp_flag, base_dir+'linkgnn/precomputation_data/'+args.dataset, seed=args.seed)
        args.num_features = train_data[0].shape[1]
        args.num_nodes = len(label)
        
    elif args.model == "DGCN":
        edge_index, features, split_idx, label = load_graph(args, base_dir, seed=args.seed)
        args.num_nodes = features.shape[0]
        args.num_features = features.shape[1]
        edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(edge_index, args.num_nodes, None)
        # data.edge_index, edge_in, in_weight, edge_out, out_weight = F_in_out(data.edge_index, data.y.size(-1), data.edge_weight)
        edge_index, features = edge_index.to(device), features.to(device)
        edge_in, in_weight, edge_out, out_weight = edge_in.to(device), in_weight.to(device), edge_out.to(device), out_weight.to(device)
    elif args.model in ["GloGNN","LINK","LINKX"]:
        edge_index, features, split_idx, label = load_graph(args, base_dir, seed=args.seed)
        args.num_nodes = features.shape[0]
        args.num_features = features.shape[1]
        if not args.directed:
            edge_index = to_undirected(edge_index)
        row, col = edge_index
        A = SparseTensor(row=row,col=col,sparse_sizes=(args.num_nodes, args.num_nodes))
        if args.model == "GloGNN":
            A = A.to_torch_sparse_coo_tensor()
            features = features.to(device)
            features = features.to(torch.float64)
            A = A.to(device)
            A = A.to(torch.float64)
        else:
            A_train = A[split_idx['train']]#.to_torch_sparse_coo_tensor()
            A_valid = A[split_idx['valid']]#.to_torch_sparse_coo_tensor()
            A_test = A[split_idx['test']]#.to_torch_sparse_coo_tensor()

    elif args.model == "ACMGCN":
        edge_index, features, split_idx, label = load_graph(args, base_dir, seed=args.seed)
        args.num_nodes = features.shape[0]
        args.num_features = features.shape[1]
        edge_index = to_undirected(edge_index)
        adj_low = to_scipy_sparse_matrix(edge_index)
        ## row normalization ##
        adj_low = sp.coo_matrix(adj_low)
        adj_low = adj_low + sp.eye(adj_low.shape[0])
        adj_low = sk_normalize(adj_low, norm='l1', axis=1)
        adj_high = -adj_low + sp.eye(adj_low.shape[0])        
        adj_low = sparse_mx_to_torch_sparse_tensor(adj_low)
        adj_high = sparse_mx_to_torch_sparse_tensor(adj_high)
        features = features.to(device)
        adj_low = adj_low.to(device)
        adj_high = adj_high.to(device)
    elif args.model in ["Digraph", "DigraphIB", "Magnet"]:
        edge_index, features, split_idx, label = load_graph(args, base_dir, seed=args.seed)
        args.num_nodes = features.shape[0]
        args.num_features = features.shape[1]
        args.features = features
        
    # print(split_idx['train'])
    train_labels = label[split_idx['train']].reshape(-1).long() # .to(device)
    valid_labels = label[split_idx['valid']].reshape(-1).long() #.to(device)
    test_labels = label[split_idx['test']].reshape(-1).long() #.to(device)
    args.C = max(int(train_labels.max()), int(valid_labels.max()), int(test_labels.max())) + 1
    
    ## extract few labels
    # train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_labels = extract_few_labels(args.label_budget, train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_labels)
   
    train_labels = train_labels.reshape(-1).long().to(device)
    valid_labels = valid_labels.reshape(-1).long().to(device)
    test_labels = test_labels.reshape(-1).long().to(device)
    
    time_preprocess = time.time() - t_start
    
    if args.dataset in ['wiki']:
        # if not (args.wo_directed or args.wo_undirected or args.wo_agg or args.wo_adj):
        # args.batch_split = wiki_batch_split
        args.batch_split = 20        
    if args.minibatch:
        # batch_size = args.batch_size
        batch_size = int(args.num_nodes / args.batch_split) + 1
        print("### Mini-Batch Training! ###")
    else:
        batch_size = args.batch_size
    
    if args.optuna:
        if args.model in ["simplink","MLP", "SGC", "FSGNN"]:
            list_bat_train = create_batch(train_data)
            list_bat_val = create_batch(valid_data)
            list_bat_test = create_batch(test_data)
        elif args.model == "LINKX" or args.model == "LINK":
            list_bat_train = create_batch([features[split_idx['train']]])
            list_bat_val = create_batch([features[split_idx['valid']]])
            list_bat_test = create_batch([features[split_idx['test']]])
        else:
            list_bat_train = [[0,len(train_labels)]]
            list_bat_val = [[0,len(valid_labels)]]
            list_bat_test = [[0,len(test_labels)]]


        def objective(trial):
            # if args.dataset in ['flickr','reddit','ogbn-products','ogbn-papers100M']:
            #     with open(base_dir+'config/search_space_large/'+args.model+'.json') as f:
            #         param = json.load(f)
            # else:
            with open(base_dir+'config/search_space/'+args.model+'.json') as f:
                param = json.load(f)
            if args.model == "MLP":
                args.weight_decay = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                nlayer = param["layer"][trial.suggest_int("layer", 0, len(param["layer"])-1, 1)]
                args.hidden = param["hidden"][trial.suggest_int("hidden", 0, len(param["hidden"])-1, 1)]
                dropout = param["dropout"][trial.suggest_int("dropout", 0, len(param["dropout"])-1, 1)]
                model = MLP_minibatch(nfeat=args.num_features,
                               nclass=args.C,
                               nhidden=args.hidden,
                               nlayer=nlayer,
                               dropout=dropout
                              ).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':args.lr,
                                   'weight_decay': args.weight_decay}]
            elif args.model == "GloGNN":
                wd = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                if args.dataset == "snap-patents":
                    # args.hidden = param["hidden"][trial.suggest_int("hidden", 0, 0, 1)]
                    args.hidden = 64
                elif args.dataset == "pokec":
                    args.hidden = param["hidden"][trial.suggest_int("hidden", 0, len(param["hidden"])-2, 1)]
                else:
                    args.hidden = param["hidden"][trial.suggest_int("hidden", 0, len(param["hidden"])-1, 1)]

                args.dropout = param["dropout"][trial.suggest_int("dropout", 0, len(param["dropout"])-1, 1)]
                args.glognn_alpha = param["glognn_alpha"][trial.suggest_int("glognn_alpha", 0, len(param["glognn_alpha"])-1, 1)]
                args.glognn_beta1 = param["glognn_beta1"][trial.suggest_int("glognn_beta1", 0, len(param["glognn_beta1"])-1, 1)]
                args.glognn_beta2 = param["glognn_beta2"][trial.suggest_int("glognn_beta2", 0, len(param["glognn_beta2"])-1, 1)]
                args.glognn_gamma = param["glognn_gamma"][trial.suggest_int("glognn_gamma", 0, len(param["glognn_gamma"])-1, 1)]
                args.norm_layers = param["norm_layers"][trial.suggest_int("norm_layers", 0, len(param["norm_layers"])-1, 1)]
                args.orders = param["orders"][trial.suggest_int("orders", 0, len(param["orders"])-1, 1)]
                args.norm_func_id = 2
                args.orders_func_id = 2
                model = GloGNN(args).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':lr,
                                   'weight_decay': wd}]
            elif args.model == "GCN":
                args.weight_decay = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                len_hidden = len(param["hidden"])
                if args.dataset in ["pokec", "snap-patent"]:
                    len_hidden -= 1
                args.hidden = param["hidden"][trial.suggest_int("hidden", 0, len_hidden-1, 1)]
                args.dropout = param["dropout"][trial.suggest_int("dropout", 0, len(param["dropout"])-1, 1)]
                model = GCN(args).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':args.lr,
                                   'weight_decay': args.weight_decay}]
            elif args.model == 'GPRGNN':
                args.weight_decay = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                args.lr_att = param["lr_att"][trial.suggest_int("lr_att", 0, len(param["lr_att"])-1, 1)]
                args.dprate = param["dp"][trial.suggest_int("dp", 0, len(param["dp"])-1, 1)]
                args.alpha = param["alpha"][trial.suggest_int("alpha", 0, len(param["alpha"])-1, 1)]
                model = GPRGNN(args).to(device)
                optimizer_sett = [{'params': model.lin1.parameters(),
                                   'weight_decay': args.weight_decay, 'lr': args.lr},
                                  {'params': model.lin2.parameters(),
                                   'weight_decay': args.weight_decay, 'lr': args.lr},
                                  {'params': model.prop1.parameters(),
                                   'weight_decay': 0.0, 'lr': args.lr_att}]
            elif args.model == "FSGNN":
                wd1 = trial.suggest_int("wd1", 0, len(param["wd1"])-1, 1)
                wd2 = trial.suggest_int("wd2", 0, len(param["wd2"])-1, 1)
                wd3 = trial.suggest_int("wd3", 0, len(param["wd3"])-1, 1)

                lr1 = trial.suggest_int("lr1", 0, len(param["lr1"])-1, 1)
                lr2 = trial.suggest_int("lr2", 0, len(param["lr2"])-1, 1)
                lr3 = trial.suggest_int("lr3", 0, len(param["lr3"])-1, 1)

                wd_att = trial.suggest_int("wd_att", 0, len(param["wd_att"])-1, 1)
                lr_att = trial.suggest_int("lr_att", 0, len(param["lr_att"])-1, 1)

                dp1 = trial.suggest_int("dropout1", 0, len(param["dropout1"])-1, 1)
                dp2 = trial.suggest_int("dropout2", 0, len(param["dropout2"])-1, 1)

                model = FSGNN_Large(nfeat=args.num_features,
                            nlayers=2*args.layer + 1,
                            nhidden=args.hidden,
                            nclass=args.C,
                            dp1=param["dropout2"][dp1],dp2=param["dropout2"][dp2]).to(device)
                optimizer_sett = [
                    {'params': model.wt1.parameters(), 'weight_decay': param["wd1"][wd1], 'lr': param["lr1"][lr1]},
                    {'params': model.fc2.parameters(), 'weight_decay': param["wd2"][wd2], 'lr': param["lr2"][lr2]},
                    {'params': model.fc3.parameters(), 'weight_decay': param["wd3"][wd3], 'lr': param["lr3"][lr3]},
                    {'params': model.att, 'weight_decay': param["wd_att"][wd_att], 'lr': param["lr_att"][lr_att]},
                    ]
            elif args.model == "SGC":
                wd = trial.suggest_int("wd", 0, len(param["wd"])-1, 1)
                lr = trial.suggest_int("lr", 0, len(param["lr"])-1, 1)
                model = SGC(nfeat=args.num_features,
                           nclass=args.C).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':param["lr"][lr],
                                   'weight_decay': param["wd"][wd]}]
            elif args.model == "DGCN":
                wd = 5e-4
                lr = trial.suggest_int("lr", 0, len(param["lr"])-1, 1)
                args.dropout = param["dropout"][trial.suggest_int("dropout", 0, len(param["dropout"])-1, 1)]
                args.hidden = param["num_filter"][trial.suggest_int("num_filter", 0, len(param["num_filter"])-1, 1)]

                model = DGCN(args).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':param["lr"][lr],
                                   'weight_decay': wd}]
                
            elif args.model == "ACMGCN":
                wd = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                args.hidden = param["hidden"][trial.suggest_int("hidden", 0, len(param["hidden"])-1, 1)]
                args.dropout = param["dropout"][trial.suggest_int("dropout", 0, len(param["dropout"])-1, 1)]
                args.layer = 2
                model = ACMGCN(args).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':lr,
                                   'weight_decay': wd}]
            elif args.model == "Magnet":
                args.weight_decay = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                args.q = param["q"][trial.suggest_int("q", 0, len(param["q"])-1, 1)]
                args.num_filter = param["num_filter"][trial.suggest_int("num_filter", 0, len(param["num_filter"])-1, 1)]
                args.dropout = 0.5
                args.K = 1
                args.layer = 2
                args.epochs = 3000
                f_node = edge_index[0]
                e_node = edge_index[1]
                L = hermitian_decomp_sparse(f_node, e_node, args.num_nodes, 
                                            args.q, 
                                            norm=True,
                                            laplacian=True, 
                                            max_eigen = 2.0, 
                                            gcn_appr = False, 
                                            edge_weight = None)
                L = cheb_poly_sparse(L, args.K)
                args.L_img = []
                args.L_real = []
                for i in range(len(L)):
                    args.L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device) )
                    args.L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device) )
                args.X_img  = torch.FloatTensor(args.features).to(device)
                args.X_real = torch.FloatTensor(args.features).to(device)
                features = args.features.to(device)
                model = Magnet(args).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':args.lr,
                                   'weight_decay': args.weight_decay}]
            elif args.model in ["Digraph","DigraphIB"]:
                args.weight_decay = 5e-4
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                args.digraph_alpha = param["digraph_alpha"][trial.suggest_int("digraph_alpha", 0, len(param["digraph_alpha"])-1, 1)]
                args.num_filter = param["num_filter"][trial.suggest_int("num_filter", 0, len(param["num_filter"])-1, 1)]
                args.dropout = 0.5
                args.K = 1
                args.layer = 2
                args.epochs = 3000
                edge_index1, edge_weights1 = get_appr_directed_adj(args.digraph_alpha, 
                                                                   edge_index.long(),
                                                                   args.num_nodes,
                                                                   args.features.dtype,
                                                                   None
                                                                   # data.edge_weight
                                                                  )
                edge_index1 = edge_index1.to(device)
                edge_weights1 = edge_weights1.to(device)
                args.features = args.features.to(device)
                if args.model == "Digraph":
                    args.edges = edge_index1
                    args.edge_weight = edge_weights1
                    del edge_index1, edge_weights1
                    model = DiModel(args).to(device)
                elif args.model == "DigraphIB":
                    edge_index2, edge_weights2 = get_second_directed_adj(edge_index.long(), 
                                                                         args.num_nodes,
                                                                         args.features.dtype, 
                                                                         None
                                                                         # data.edge_weight
                                                                         )
                    edge_index2 = edge_index2.to(device)
                    edge_weights2 = edge_weights2.to(device)
                    args.edges = (edge_index1, edge_index2)
                    args.edge_weight = (edge_weights1, edge_weights2)
                    del edge_index2, edge_weights2
                    model = DiGCN_IB(args).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':args.lr,
                                   'weight_decay': args.weight_decay}]
            if args.model == 'simplink':
                with open(base_dir+'config/search_space/simplink.json') as f:
                    param = json.load(f)
                args.weight_decay = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                args.A_weight_decay = param["wd"][trial.suggest_int("A_wd", 0, len(param["wd"])-1, 1)]
                args.X_weight_decay = param["wd"][trial.suggest_int("X_wd", 0, len(param["wd"])-1, 1)]
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                args.A_lr = param["lr"][trial.suggest_int("A_lr", 0, len(param["lr"])-1, 1)]
                args.X_lr = param["lr"][trial.suggest_int("X_lr", 0, len(param["lr"])-1, 1)]
                # args.linkx_hidden = param["hidden"][trial.suggest_int("linkx_hidden", 0, len(linkx_param["hidden"])-1, 1)]
                args.dropout = param["dropout"][trial.suggest_int("dropout", 0, len(param["dropout"])-1, 1)]
                
                if not args.wo_att:                
                    args.wd_att = param["wd_att"][trial.suggest_int("wd_att", 0, len(param["wd_att"])-1, 1)]
                    args.lr_att = param["lr_att"][trial.suggest_int("lr_att", 0, len(param["lr_att"])-1, 1)]
                
                args.num_edge_layers = param["num_edge_layers"][trial.suggest_int("num_edge_layers", 0, len(param["num_edge_layers"])-1, 1)]
                args.num_node_layers = param["num_node_layers"][trial.suggest_int("num_node_layers", 0, len(param["num_node_layers"])-1, 1)]
                args.num_agg_layers = param["num_agg_layers"][trial.suggest_int("num_agg_layers", 0, len(param["num_agg_layers"])-1, 1)]
                args.final_layers = param["layers"][trial.suggest_int("layers", 0, len(param["layers"])-1, 1)]
                args.final_weight_decay = param["wd"][trial.suggest_int("final_wd", 0, len(param["wd"])-1, 1)]
                args.final_lr = param["lr"][trial.suggest_int("final_lr", 0, len(param["lr"])-1, 1)]
                # print(args.weight_decay,args.lr,args.dropout, args.num_edge_layers, args.num_node_layers, args.layers)
                
                # args.linkgnn_wd_att = trial.suggest_int("linkgnn_wd_att", 0, len(linkx_param["linkgnn_wd_att"])-1, 1)
                # model = LINKGNN(args, GNN_undi, GNN_di, GNN_di_t).to(device)
                model = simplink(args, args.layer).to(device)
                optimizer_sett = [
                    # {'params': model.wt1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                   # {'params': model.W.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr},
                    {'params': model.mlp_final.parameters(), 'weight_decay': args.final_weight_decay, 'lr': args.final_lr}]
                if not args.wo_att:
                    optimizer_sett += [
                        {'params': model.att, 'weight_decay': args.wd_att, 'lr': args.lr_att}
                    ]
                if not args.wo_mlp:
                    optimizer_sett += [
                        {'params': model.mlpX.parameters(), 'weight_decay': args.X_weight_decay, 'lr': args.X_lr}
                    ]
                if not args.wo_adj:
                    if not args.wo_undirected:
                        optimizer_sett += [
                            {'params': model.mlpA.parameters(), 'weight_decay': args.A_weight_decay, 'lr': args.A_lr},
                        ]
                    if not args.wo_directed:
                        optimizer_sett += [
                            {'params': model.mlpA_di.parameters(), 'weight_decay': args.A_weight_decay, 'lr': args.A_lr},
                        ]
                        if not args.wo_transpose:
                            optimizer_sett += [
                                {'params': model.mlpA_di_t.parameters(), 'weight_decay': args.A_weight_decay, 'lr': args.A_lr},
                            ]
                if not args.wo_agg:
                    if not args.wo_undirected:
                        optimizer_sett += [
                            {'params': model.mlp_agg.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                        ]
                    if not args.wo_directed:
                        optimizer_sett += [
                            {'params': model.mlp_agg_di.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},                 
                        ]
                        if not args.wo_transpose:
                            optimizer_sett += [
                                {'params': model.mlp_agg_di_t.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                            ]
            elif args.model == "LINKX":
                args.weight_decay = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                args.hidden = param["hidden"][trial.suggest_int("hidden", 0, len(param["hidden"])-1, 1)]
                args.dropout = param["dropout"][trial.suggest_int("dropout", 0, len(param["dropout"])-1, 1)]
                args.num_edge_layers = param["num_edge_layers"][trial.suggest_int("num_edge_layers", 0, len(param["num_edge_layers"])-1, 1)]
                args.num_node_layers = param["num_node_layers"][trial.suggest_int("num_node_layers", 0, len(param["num_node_layers"])-1, 1)]
                args.layers = param["layers"][trial.suggest_int("layers", 0, len(param["layers"])-1, 1)]
                model = LINKX(args).to(device)
                print(args.weight_decay,args.lr,args.dropout, args.num_edge_layers, args.num_node_layers, args.layers)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':args.lr,
                                   'weight_decay': args.weight_decay}]
            elif args.model == "LINK":
                args.weight_decay = param["wd"][trial.suggest_int("wd", 0, len(param["wd"])-1, 1)]
                args.lr = param["lr"][trial.suggest_int("lr", 0, len(param["lr"])-1, 1)]
                args.hidden = param["hidden"][trial.suggest_int("hidden", 0, len(param["hidden"])-1, 1)]
                args.num_edge_layers = param["num_edge_layers"][trial.suggest_int("num_edge_layers", 0, len(param["num_edge_layers"])-1, 1)]
                model = LINK(args).to(device)
                optimizer_sett = [{'params':model.parameters(),
                                   'lr':args.lr,
                                   'weight_decay': args.weight_decay}]

                
            if args.adam:
                optimizer = optim.Adam(optimizer_sett)
            else:
                optimizer = optim.AdamW(optimizer_sett)
            
            best_dic = train_step(model, optimizer)
            print("\nbest : val acc =", best_dic['val_result']['accuracy'])
            return best_dic['val_result']['accuracy']
            # return best_dic['val_loss']

        t_total = time.time()

        # optimization
        study = optuna.create_study(direction="maximize")
        # study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        with open(base_dir+"config/best_config_optuna/"+linkgnn_flag+args.model+ablation_tag+"_"+args.dataset+"_budget"+str(args.label_budget)+"_best_params.json","w") as f:
            json.dump(study.best_params,f,indent=4)
        print("\nbest : params / value =  " , study.best_params, " / ", study.best_value)
                    
    if args.test:
        # if args.dataset in ['flickr','reddit','ogbn-products','ogbn-papers100M']:
        #     with open(base_dir+'config/search_space_large/'+args.model+'.json') as f:
        #         param = json.load(f)
        # else:
        with open(base_dir+'config/search_space/'+args.model+'.json') as f:
            param = json.load(f)
        with open(base_dir+"config/best_config_optuna/"+linkgnn_flag+args.model+ablation_tag+"_"+args.dataset+"_budget"+str(args.label_budget)+"_best_params.json") as f:
            best_param = json.load(f)
        
        seed = args.seed
        res_dic = dict()
        if args.model in ["simplink","MLP", "SGC", "FSGNN"]:
            list_bat_train = create_batch(train_data)
            list_bat_val = create_batch(valid_data)
            list_bat_test = create_batch(test_data)
        elif args.model == "LINKX" or args.model == "LINK":
            list_bat_train = create_batch([features[split_idx['train']]])
            list_bat_val = create_batch([features[split_idx['valid']]])
            list_bat_test = create_batch([features[split_idx['test']]])
        else:
            list_bat_train = [[0,len(train_labels)]]
            list_bat_val = [[0,len(valid_labels)]]
            list_bat_test = [[0,len(test_labels)]]
            # list_bat_train = [[0,args.num_nodes]]
            # list_bat_val = [[0,args.num_nodes]]
            # list_bat_test = [[0,args.num_nodes]]
            
        if args.model == "MLP":
            model = MLP_minibatch(nfeat=args.num_features,
                           nclass=args.C,
                           nhidden=param["hidden"][best_param["hidden"]],
                           nlayer=param["layer"][best_param["layer"]],
                           dropout=param["dropout"][best_param["dropout"]]
                          ).to(device)
            optimizer_sett = [{'params': model.parameters(),
                               'lr': param["lr"][best_param["lr"]],
                               'weight_decay': param["wd"][best_param["wd"]]}]
        elif args.model == "GloGNN":
            wd = param["wd"][best_param["wd"]]
            lr = param["lr"][best_param["lr"]]
            if args.dataset == "snap-patents":
                args.hidden = 64
            else:
                args.hidden = param["hidden"][best_param["hidden"]]
            args.dropout = param["dropout"][best_param["dropout"]]
            args.glognn_alpha = param["glognn_alpha"][best_param["glognn_alpha"]]
            args.glognn_beta1 = param["glognn_beta1"][best_param["glognn_beta1"]]
            args.glognn_beta2 = param["glognn_beta2"][best_param["glognn_beta2"]]
            args.glognn_gamma = param["glognn_gamma"][best_param["glognn_gamma"]]
            args.norm_layers = param["norm_layers"][best_param["norm_layers"]]
            args.orders = param["orders"][best_param["orders"]]
            args.norm_func_id = 2
            args.orders_func_id = 2
            model = GloGNN(args).to(device)
            optimizer_sett = [{'params':model.parameters(),
                               'lr':lr,
                               'weight_decay': wd}]
        elif args.model == "GCN":
            args.weight_decay = param['wd'][best_param['wd']]
            args.lr = param['lr'][best_param['lr']]
            args.dropout = param['dropout'][best_param['dropout']]
            model = GCN(args).to(device)
            optimizer_sett = [{'params':model.parameters(),
                               'lr':args.lr,
                               'weight_decay':args.weight_decay}]
        elif args.model == 'GPRGNN':
            args.weight_decay = param['wd'][best_param['wd']]
            args.lr = param['lr'][best_param['lr']]
            args.dprate = param["dp"][best_param["dp"]]
            args.alpha = param["alpha"][best_param["alpha"]]
            args.lr_att = param["lr_att"][best_param["lr_att"]]
            model = GPRGNN(args).to(device)
            optimizer_sett = [{'params': model.lin1.parameters(),
                               'weight_decay': args.weight_decay, 'lr': args.lr},
                              {'params': model.lin2.parameters(),
                               'weight_decay': args.weight_decay, 'lr': args.lr},
                              {'params': model.prop1.parameters(),
                               'weight_decay': 0.0, 'lr': args.lr_att}]
        elif args.model == "DGCN":
            wd = 5e-4
            lr = param["lr"][best_param["lr"]]
            args.dropout = param["dropout"][best_param["dropout"]]
            args.hidden = param["num_filter"][best_param["num_filter"]]

            model = DGCN(args).to(device)
            optimizer_sett = [{'params':model.parameters(),
                               'lr':lr,
                               'weight_decay': wd}]
        elif args.model == "ACMGCN":
            wd = param["wd"][best_param["wd"]]
            lr = param["lr"][best_param["lr"]]
            args.hidden = param["hidden"][best_param["hidden"]]
            args.dropout = param["dropout"][best_param["dropout"]]
            args.layer = 2
            model = ACMGCN(args).to(device)
            optimizer_sett = [{'params':model.parameters(),
                               'lr':lr,
                               'weight_decay': wd}]
        elif args.model == "FSGNN":
            layer_norm = bool(int(args.layer_norm))
            model = FSGNN_Large(nfeat=args.num_features,
                                nlayers=2*args.layer + 1,
                                nhidden=args.hidden,
                                nclass=args.C,
                                dp1=param["dropout1"][best_param["dropout1"]],
                                dp2=param["dropout2"][best_param["dropout2"]]).to(device)
            optimizer_sett = [
                {'params': model.wt1.parameters(), 'weight_decay': param["wd1"][best_param["wd1"]], 'lr': param["lr1"][best_param["lr1"]]},
                {'params': model.fc2.parameters(), 'weight_decay': param["wd2"][best_param["wd2"]], 'lr': param["lr2"][best_param["lr2"]]},
                {'params': model.fc3.parameters(), 'weight_decay': param["wd3"][best_param["wd3"]], 'lr': param["lr3"][best_param["lr3"]]},
                {'params': model.att, 'weight_decay': param["wd_att"][best_param["wd_att"]], 'lr': param["lr_att"][best_param["lr_att"]]},
                ]
        elif args.model == "SGC":
            model = SGC(nfeat=args.num_features,
                        nclass=args.C,
                        ).to(device)
            optimizer_sett = [{'params': model.parameters(),
                               'lr': param["lr"][best_param["lr"]],
                               'weight_decay': param["wd"][best_param["wd"]]}]
        elif args.model == 'LINKX':
            # args.weight_decay = param['wd'][best_param['wd']]
            # args.lr = param['lr'][best_param['lr']]
            args.hidden = param["hidden"][best_param["hidden"]]
            args.num_edge_layers = param["num_edge_layers"][best_param["num_edge_layers"]]
            args.num_node_layers = param["num_node_layers"][best_param["num_node_layers"]]
            args.layers = param["layers"][best_param["layers"]]
            model = LINKX(args).to(device)
            optimizer_sett = [{'params': model.parameters(),
                               'lr': param["lr"][best_param["lr"]],
                               'weight_decay': param["wd"][best_param["wd"]]}]
        elif args.model == 'LINK':
            args.hidden = param["hidden"][best_param["hidden"]]
            args.num_edge_layers = param["num_edge_layers"][best_param["num_edge_layers"]]
            # args.layers = param["layers"][best_param["layers"]]
            model = LINK(args).to(device)
            optimizer_sett = [{'params': model.parameters(),
                               'lr': param["lr"][best_param["lr"]],
                               'weight_decay': param["wd"][best_param["wd"]]}]
        elif args.model == "Magnet":
            args.weight_decay = param["wd"][best_param["wd"]]
            args.lr = param["lr"][best_param["lr"]]
            args.q = param["q"][best_param["q"]]
            args.num_filter = param["num_filter"][best_param["num_filter"]]
            args.dropout = 0.5
            args.K = 1
            args.layer = 2
            args.epochs = 3000
            f_node = edge_index[0]
            e_node = edge_index[1]
            L = hermitian_decomp_sparse(f_node, e_node, args.num_nodes, 
                                        args.q, 
                                        norm=True,
                                        laplacian=True, 
                                        max_eigen = 2.0, 
                                        gcn_appr = False, 
                                        edge_weight = None)
            L = cheb_poly_sparse(L, args.K)
            args.L_img = []
            args.L_real = []
            for i in range(len(L)):
                args.L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device) )
                args.L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device) )
            args.X_img  = torch.FloatTensor(features).to(device)
            args.X_real = torch.FloatTensor(features).to(device)
            features = features.to(device)
            model = Magnet(args).to(device)
            optimizer_sett = [{'params':model.parameters(),
                               'lr':args.lr,
                               'weight_decay': args.weight_decay}]
        elif args.model in ["Digraph","DigraphIB"]:
            args.weight_decay = 5e-4
            args.lr = param["lr"][best_param["lr"]]
            args.digraph_alpha = param["digraph_alpha"][best_param["digraph_alpha"]]
            args.num_filter = param["num_filter"][best_param["num_filter"]]
            args.dropout = 0.5
            args.K = 1
            args.layer = 2
            args.epochs = 3000
            edge_index1, edge_weights1 = get_appr_directed_adj(args.digraph_alpha, 
                                                               edge_index.long(),
                                                               args.num_nodes,
                                                               args.features.dtype,
                                                               None
                                                               # data.edge_weight
                                                              )
            edge_index1 = edge_index1.to(device)
            edge_weights1 = edge_weights1.to(device)
            args.features = args.features.to(device)
            if args.model == "Digraph":
                args.edges = edge_index1
                args.edge_weight = edge_weights1
                del edge_index1, edge_weights1
                model = DiModel(args).to(device)
            elif args.model == "DigraphIB":
                edge_index2, edge_weights2 = get_second_directed_adj(edge_index.long(), 
                                                                     args.num_nodes,
                                                                     args.features.dtype, 
                                                                     None
                                                                     # data.edge_weight
                                                                     )
                edge_index2 = edge_index2.to(device)
                edge_weights2 = edge_weights2.to(device)
                args.edges = (edge_index1, edge_index2)
                args.edge_weight = (edge_weights1, edge_weights2)
                del edge_index2, edge_weights2
                model = DiGCN_IB(args).to(device)
            optimizer_sett = [{'params': model.parameters(),
                               'lr': args.lr,
                               'weight_decay': args.weight_decay}]
        elif args.model == 'simplink':
            # with open(base_dir+'config/search_space/simplink.json') as f:
            #     param = json.load(f)
            args.weight_decay = param["wd"][best_param["wd"]]
            args.A_weight_decay = param["wd"][best_param["A_wd"]]
            args.X_weight_decay = param["wd"][best_param["X_wd"]]
            args.lr = param["lr"][best_param["lr"]]
            args.A_lr = param["lr"][best_param["A_lr"]]
            args.X_lr = param["lr"][best_param["X_lr"]]
            # args.linkx_hidden = param["hidden"][trial.suggest_int("linkx_hidden", 0, len(linkx_param["hidden"])-1, 1)]
            args.dropout = param["dropout"][best_param["dropout"]]

            if not args.wo_att:
                args.wd_att = param["wd_att"][best_param["wd_att"]]
                args.lr_att = param["lr_att"][best_param["lr_att"]]

            args.num_edge_layers = param["num_edge_layers"][best_param["num_edge_layers"]]
            args.num_node_layers = param["num_node_layers"][best_param["num_node_layers"]]
            args.num_agg_layers = param["num_agg_layers"][best_param["num_agg_layers"]]
            args.final_layers = param["layers"][best_param["layers"]]
            args.final_weight_decay = param["wd"][best_param["final_wd"]]
            args.final_lr = param["lr"][best_param["final_lr"]]
            # args.linkgnn_wd_att = linkx_param["linkgnn_wd_att"][best_param["linkgnn_wd_att"]]
            # print(args.weight_decay,args.lr,args.dropout, args.num_edge_layers, args.num_node_layers, args.layers)
            # model = LINKGNN(args, GNN_undi, GNN_di, GNN_di_t).to(device)
            model = simplink(args, args.layer).to(device)
            optimizer_sett = [
                               # {'params': model.wt1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                               # {'params': model.W.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr},
                {'params': model.mlp_final.parameters(), 'weight_decay': args.final_weight_decay, 'lr': args.final_lr}]
            if not args.wo_att:
                optimizer_sett += [
                    {'params': model.att, 'weight_decay': args.wd_att, 'lr': args.lr_att}
                ]
            if not args.wo_mlp:
                optimizer_sett += [
                    {'params': model.mlpX.parameters(), 'weight_decay': args.X_weight_decay, 'lr': args.X_lr}
                ]
            if not args.wo_adj:
                if not args.wo_undirected:
                    optimizer_sett += [
                        {'params': model.mlpA.parameters(), 'weight_decay': args.A_weight_decay, 'lr': args.A_lr},
                    ]
                if not args.wo_directed:
                    optimizer_sett += [
                        {'params': model.mlpA_di.parameters(), 'weight_decay': args.A_weight_decay, 'lr': args.A_lr},
                    ]
                    if not args.wo_transpose:
                        optimizer_sett += [
                            {'params': model.mlpA_di_t.parameters(), 'weight_decay': args.A_weight_decay, 'lr': args.A_lr},
                        ]
            if not args.wo_agg:
                if not args.wo_undirected:
                    optimizer_sett += [
                        {'params': model.mlp_agg.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                    ]
                if not args.wo_directed:
                    optimizer_sett += [
                        {'params': model.mlp_agg_di.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},                 
                    ]
                    if not args.wo_transpose:
                        optimizer_sett += [
                            {'params': model.mlp_agg_di_t.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
                        ]

        if args.linkgnn and args.model != 'simplink':
            with open(base_dir+'config/search_space/LINKGNN.json') as f:
                linkx_param = json.load(f)
            args.linkx_weight_decay = linkx_param["wd"][best_param["linkx_wd"]]
            args.linkx_lr = linkx_param["lr"][best_param["linkx_lr"]]
            # args.linkx_hidden = param["hidden"][trial.suggest_int("linkx_hidden", 0, len(linkx_param["hidden"])-1, 1)]
            args.linkx_dropout = linkx_param["dropout"][best_param["linkx_dropout"]]
            args.num_edge_layers = linkx_param["num_edge_layers"][best_param["num_edge_layers"]]
            args.num_node_layers = linkx_param["num_node_layers"][best_param["num_node_layers"]]
            args.linkx_layers = linkx_param["layers"][best_param["linkx_layers"]]
            args.linkgnn_wd_att = linkx_param["linkgnn_wd_att"][best_param["linkgnn_wd_att"]]
            # print(args.weight_decay,args.lr,args.dropout, args.num_edge_layers, args.num_node_layers, args.layers)
            model = LINKGNN(args, GNN_undi, GNN_di, GNN_di_t).to(device)
            optimizer_sett = optimizer_sett + [{'params': model.mlpA.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr},
                                               {'params': model.mlpA_di.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr},
                                               {'params': model.mlpA_di_t.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr},
                                               {'params': model.mlpX.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr},
                                               # {'params': model.W.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr},
                                               {'params': model.att, 'weight_decay': args.linkgnn_wd_att, 'lr': 0.01},
                                               {'params': model.mlp_final.parameters(), 'weight_decay': args.linkx_weight_decay, 'lr': args.linkx_lr}]

        if args.adam:
            optimizer = optim.Adam(optimizer_sett)
        else:
            optimizer = optim.AdamW(optimizer_sett)
        t_start = time.time()
        # res_dic["val_acc"], best, res_dic["best_epoch"], best_model, res_dic["train_acc"], res_dic["epochs_stopped"] 
        res_dic = train_step(model,optimizer)
        res_dic["train_time"] = time.time()-t_start
        res_dic["preprocess_time"] = time_preprocess

        # test_num = test_data[0][0].shape[0]
        # test_num = test_labels.shape[0]
        # test_data = [mat.to(device) for mat in test_data[:total_filt]]
        torch.cuda.empty_cache()
        if args.test:
            list_loss_test = []
            pred_test = torch.tensor([])
            out_AUC_test = []
            label_test = torch.tensor([])
            model = res_dic['best_model']
            for st,end in list_bat_test:
                loss_test, pred, out_AUC = test(model,st,end)
                pred_test = torch.concat([pred_test,pred.detach().to('cpu')],axis=0)
                if len(out_AUC_test) == 0:
                    out_AUC_test = out_AUC.detach().to('cpu')
                else:
                    out_AUC_test = torch.concat([out_AUC_test,out_AUC.detach().to('cpu')],axis=0)
                label_test = torch.concat([label_test,test_labels[st:end].to('cpu')],axis=0)
                list_loss_test.append(loss_test)

            loss_test = np.mean(list_loss_test)
            res_dic['test_result'] = calc_metric(pred_test.numpy(),label_test.numpy(),out_AUC_test.numpy())


        res_dic.pop('best_model')
        if not os.path.exists(base_dir+"experiments/result_optuna/"+args.dataset):
            os.mkdir(base_dir+"experiments/result_optuna/"+args.dataset)
        with open(base_dir+"experiments/result_optuna/"+args.dataset+"/"+linkgnn_flag+args.model+ablation_tag+"_budget"+str(args.label_budget)+"_seed"+str(seed)+'.json','w') as f:
            json.dump(res_dic, f, ensure_ascii=False, indent=4) 
        print("Train cost: {:.4f}s".format(res_dic["train_time"]))
        print('Load {}th epoch'.format(res_dic["best_epoch"]))

        if args.test:
            print(f"Valdiation accuracy: {np.round(res_dic['val_result']['accuracy']*100,2)}, Test accuracy: {np.round(res_dic['test_result']['accuracy']*100,2)}")

if __name__ == '__main__':
# Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='simplink',help='Model')
    parser.add_argument('--dataset',type=str,default='flickr',help='Dataset')
    parser.add_argument('--seed', type=int, default=100, help='Random seed.')
    
## parameters for training ##
    parser.add_argument('--batch_size', type=int, default= 10000000000,# 4096,
                        help='Batchsize for mini-batch training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=40,
                        help='Patience for early stop')
    parser.add_argument('--minibatch', action='store_true')
    parser.add_argument('--batch_split', type=int, default=10, help='split size for minibatch training.')
    
## parameters for models ##
### common ###
    parser.add_argument('--hidden', type=int, default=128,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate.')

### FSGNN ###
    parser.add_argument('--layer', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--layer_norm', type=int, default=1,
                        help='Layer normalization')
    parser.add_argument('--wd3', type=float, default=5e-4,
                        help='weight decay.')
    parser.add_argument('--lr3', type=float, default=0.002,
                        help='learning rate.')

### LINKX ###
    parser.add_argument('--num_edge_layers', type=int, default=2)   # LINKX
    parser.add_argument('--num_node_layers', type=int, default=2)   # LINKX
    
## GloGNN ## 
    # used for mlpnorm
    parser.add_argument('--glognn_alpha', type=float, default=0.0,
                        help='Weight for node features, thus 1-alpha for adj')
    parser.add_argument('--glognn_beta1', type=float, default=0.0,
                        help='Weight for frobenius norm on Z.')
    parser.add_argument('--glognn_beta2', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--glognn_gamma', type=float, default=0.0,
                        help='Weight for MLP results kept')
    parser.add_argument('--norm_func_id', type=int, default=2,
                        help='Function of norm layer, ids \in [1, 2]')
    parser.add_argument('--norm_layers', type=int, default=1,
                        help='Number of groupnorm layers')
    parser.add_argument('--orders_func_id', type=int, default=2,
                        help='Sum function of adj orders in norm layer, ids \in [1, 2, 3]')
    parser.add_argument('--orders', type=int, default=1,
                        help='Number of adj orders in norm layer')
    
## GloGNN and LINKX ##
    parser.add_argument('--directed', action='store_true',
                        help='If true, a given adjacency matrix is handled as a directed graph.')
    
## hardware ##
    parser.add_argument('--cuda', type=int, default=1,
                        help='GPU ID')
    
## train/tuning ##
    parser.add_argument('--optuna', action='store_true')
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--num_restart', type=int, default=5,
                        help='Number of trials')
    parser.add_argument('--adam', action='store_true')
    parser.add_argument('--label_budget', type=int, default=0, help='label budget for each class.')
    
## ablation study ##
    parser.add_argument('--agg',type=str,default='concat',help='aggregation')
    parser.add_argument('--linkgnn', action='store_true') 
    parser.add_argument('--wo_att', action='store_true')
    parser.add_argument('--wo_mlp', action='store_true')
    parser.add_argument('--wo_adj', action='store_true')
    parser.add_argument('--wo_agg', action='store_true')
    parser.add_argument('--wo_directed', action='store_true')
    parser.add_argument('--wo_undirected', action='store_true')
    parser.add_argument('--wo_transpose', action='store_true')

    args = parser.parse_args()

    fix_seed(args.seed)
    main(args)