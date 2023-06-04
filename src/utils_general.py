from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn.functional as F
import torch_sparse as ts
import numpy as np
import scipy.sparse as sp
import random
from dataset_utils import *
from torch_geometric.utils import to_undirected, sort_edge_index
import sys
import os
import pickle as pkl

sys.setrecursionlimit(99999)


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def calc_metric(out_tmp, label_tmp, out_AUC):
    import warnings
    warnings.simplefilter("ignore")
    try:
        if label_tmp.shape[1] != 1:
             label_tmp = np.argmax(label_tmp, axis=1) 
    except Exception:
        True
    tmp_dic = classification_report(label_tmp, out_tmp, output_dict=True)
    if out_AUC.shape[1] == 2:
        out_AUC = out_AUC[:,1]
        # tmp_dic["roc_auc"] = roc_auc_score(label_tmp, out_AUC, multi_class="ovr")
        tmp_dic["roc_auc"] = roc_auc_score(label_tmp, out_AUC)
    return tmp_dic

def encode_onehot(labels, num_classes):
    one_hot = torch.zeros(labels.shape[0], num_classes)
    one_hot.scatter_(1, labels.unsqueeze(-1), 1)
    return one_hot

def normalization(adj_t):
    deg = ts.sum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = ts.mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = ts.mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

def load_npz_dataset(file_name):
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        # edge_index = [torch.from_numpy(loader['adj_indices']).long(), torch.from_numpy(loader['adj_indptr']).long()]
        
        # print(loader['adj_data'])
        indices = loader['adj_indices']
        indptr = loader['adj_indptr']
        A = sp.csr_matrix((loader['adj_data'], indices,
                           indptr), shape=loader['adj_shape'])
        # csr_matrix = sp.csr_matrix((data, indices, indptr))
        rows = []
        cols = []
        for i in range(A.shape[0]):
            start = indptr[i]
            end = indptr[i + 1]
            rows.extend([i] * (end - start))
            cols.extend(indices[start:end])
        edge_index = torch.tensor([rows, cols]).long()

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])
        X = X.toarray()
        X = torch.from_numpy(X).float()
        
        labels = torch.from_numpy(loader.get('labels')).long()

        graph = {
            # 'A': A,
            'edge_index': edge_index,
            'features': X,
            'labels': labels
        }
        return graph
    
def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = torch.tensor(train_mask)
    mask['valid'] = torch.tensor(val_mask)
    mask['test'] = torch.tensor(test_mask)
    return mask

def extract_few_labels(label_budget, train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_labels, smallest_class_threshold=3):
    too_small_class = set()
    if label_budget > 0:
        few_train = torch.tensor([], dtype=torch.int64)
        for l in range(num_labels):
            tmp_class_label = torch.flatten((train_labels==l).nonzero())
            # try:
            # print(len(tmp_class_label))
            if len(tmp_class_label) < smallest_class_threshold or len(tmp_class_label) < label_budget: ## if a class is smaller than a user-specified threshold, then it will be removed.
                too_small_class.add(l)
            else:
                tmp_labels = torch.tensor(np.random.choice(tmp_class_label, label_budget, replace=False), dtype=torch.int64)
                # print(tmp_labels)
                few_train = torch.concat([few_train,tmp_labels], axis=0)

        train_labels = train_labels[few_train]
        for i in range(len(train_data)):
            train_data[i] = train_data[i][few_train]#.to(device)
    # print(few_train)
    ## re-ordering classes
    class_order_dic = dict()
    all_class = set(range(num_labels))
    remaining_class = all_class-too_small_class
    for ind, l in enumerate(sorted(list(remaining_class))):
        class_order_dic[l] = ind
    num_few_labels = len(remaining_class)
    print(too_small_class)
    print("number of classes:", num_few_labels)
        
    # tmp_train_labels.append(original_train[few_train].reshape(-1).long())

    valid_wo_small = torch.tensor([], dtype=torch.int64)
    test_wo_small = torch.tensor([], dtype=torch.int64)
    for l in range(len(all_class)):
        if l in too_small_class:
            continue
        else:
            tmp_class_val = torch.flatten((valid_labels==l).nonzero())
            valid_wo_small = torch.concat([valid_wo_small,tmp_class_val], axis=0)
            tmp_class_test = torch.flatten((test_labels==l).nonzero())
            test_wo_small = torch.concat([test_wo_small,tmp_class_test], axis=0)
            
    for i in range(len(valid_data)):
        valid_data[i] = valid_data[i][valid_wo_small]
    for i in range(len(test_data)):
        test_data[i] = test_data[i][test_wo_small]

    def re_order(in_labels, class_order_dic):
        in_labels = in_labels.numpy()
        for i in range(len(in_labels)):
            in_labels[i] = class_order_dic[in_labels[i]]
        return torch.tensor(in_labels)
    train_labels = re_order(train_labels, class_order_dic)
    valid_labels = re_order(valid_labels[valid_wo_small], class_order_dic)
    test_labels = re_order(test_labels[test_wo_small], class_order_dic)
    
    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_few_labels


    
def load_graph(args, data_root, seed=100):
    from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset
    import torch_geometric.transforms as T
    from torch_geometric.datasets import Reddit2, Flickr, WebKB, WikipediaNetwork, Actor, Amazon, Planetoid

    fix_seed(seed)
    
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
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        del data
        
    elif args.dataset in ["fb100","deezer-europe","arxiv-year","pokec","snap-patents","yelp-chi","genius","twitch-gamer","wiki"]:
        if args.dataset == "fb100":
            # if sub_dataname not in ("Penn94", "Amherst41", "Cornell5", "Johns Hopkins55", "Reed98"):
                # print("Invalid sub_dataname, deferring to Penn94 graph")
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
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        # print(train_idx)
        del dataset
    elif args.dataset in ["squirrel-filtered-directed","chameleon-filtered-directed"]:
        data_dgl = Dataset(name=args.dataset)
        edge_index =  torch.stack(list(data_dgl.graph.edges()),dim=0).long()
        # edge_index = torch.tensor([data_dgl.graph.edges()[0],data_dgl.graph.edges()[1]])
        N = data_dgl.graph.num_nodes()
        features = data_dgl.node_features
        labels = data_dgl.labels
        d = features.shape[1]
        split_idx = dict()
        split_idx["train"] = data_dgl.train_idx_list[int(seed/100)]
        split_idx["valid"] = data_dgl.val_idx_list[int(seed/100)]
        split_idx["test"] = data_dgl.test_idx_list[int(seed/100)]
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
        split_idx = dict()
        split_idx["train"] = torch.flatten((mask['train']==True).nonzero())
        split_idx["valid"] = torch.flatten((mask['valid']==True).nonzero())
        split_idx["test"] = torch.flatten((mask['test']==True).nonzero())
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
            # if args.dataset in ["cora","citeseer","pubmed"]:
            #     data_func = Planetoid
            if args.dataset in ["wisconsin", "cornell", "texas"]:
                data_func = WebKB
                split_flag = True # there are several default splits
            elif args.dataset in ["chameleon","squirrel"]:
                data_func = WikipediaNetwork
                split_flag = True # there are several default splits
            elif args.dataset in ["computers","photo"]:
                data_func = Amazon
            dataset = data_func(data_root+"dataset/"+args.dataset+"/", args.dataset)
            
            # dataset = data_func(data_root+"dataset/"+args.dataset+"/", args.dataset, transform=T.NormalizeFeatures())
            
        data = dataset[0]
        split_idx = dict()
        if split_flag == True:
            split_idx["train"] = torch.flatten((torch.t(data.train_mask)[0]==True).nonzero())
            split_idx["valid"] = torch.flatten((torch.t(data.val_mask)[0]==True).nonzero())
            split_idx["test"] = torch.flatten((torch.t(data.test_mask)[0]==True).nonzero())
        else:
            split_idx["train"] = torch.flatten((data.train_mask==True).nonzero())
            split_idx["valid"] = torch.flatten((data.val_mask==True).nonzero())
            split_idx["test"] = torch.flatten((data.test_mask==True).nonzero())
            
        edge_index = data.edge_index
        N = data.num_nodes
        labels = data.y.data
        features = data.x
        d = data.num_features
        del data
        
    # if args.dataset not in ["arxiv-year", "snap-patents", "squirrel", "chameleon"]:
    # edge_index_undi = to_undirected(edge_index)
    return edge_index, features, split_idx, labels.reshape(-1).long()
    
def load_precomp(args, precomp_flag, data_path, seed=100):
    import process_spmm_dire as process_spmm
    import pickle as pkl
    # if precomp_flag == False and (not os.path.exists(data_path+"/adjacency_di_t_3_training_"+str(seed)+".pickle") or not os.path.exists(data_path+"/exact1hop_di_t_3_training_"+str(seed)+".pickle")):
    if args.model == "MLP":
        args.layer = 0
        fliters = []
    elif args.model == "SGC":
        args.layer = 2
        filters = ["adjacency"]
    elif args.model == "FSGNN":
        filters = ["adjacency", "exact1hop"]
    elif args.model == "simplink":
        filters = ["adjacency","exact1hop","adjacency_di","exact1hop_di","adjacency_di_t","exact1hop_di_t"]
    args.inductive = False
    
    # if not args.optuna:
    for filt in filters:
        args.filter = filt
        process_spmm.main(args, seed)
        # args.filter = "adjacency"
        # args.inductive = False
        # process_spmm.main(args)
        # if args.model == "FSGNN":
        #     args.filter = "exact1hop"
        #     process_spmm.main(args)

        # if args.dataset in ["flickr", "reddit"]:
        #     args.filter = "adjacency"
        #     args.inductive = True
        #     process_spmm.main(args)
        #     if args.model == "FSGNN":
        #         args.filter = "exact1hop"
        #         process_spmm.main(args)
            
    ### Data Load ###
    # Load node features used for input model #
    if args.model in ["simplink","FSGNN","JKnet_LC","MLP"]:
        if args.model in ["simplink"]:
            # filters = ["adjacency", "exact1hop"]
            filters = ["adjacency","exact1hop","adjacency_di","exact1hop_di","adjacency_di_t","exact1hop_di_t"]
        elif args.model == "FSGNN":
            filters = ["adjacency","exact1hop"]
        elif args.model == "JKnet_LC":
            filters = ["adjacency"]
        elif args.model == "MLP":
            filters = []
        # training data
        train_data = []
        if args.model != "simplink":
            with open(data_path+"/feature_training_"+str(seed)+".pickle","rb") as fopen:
                train_data.append(pkl.load(fopen))
        for filt in filters:
            for i in range(1,args.layer+1):
                with open(data_path+"/"+filt+"_"+str(i)+"_training_"+str(seed)+".pickle","rb") as fopen:
                    train_data.append(pkl.load(fopen))
        # validation data
        valid_data = []
        if args.model != "simplink":
            with open(data_path+"/feature_validation_"+str(seed)+".pickle","rb") as fopen:
                valid_data.append(pkl.load(fopen))
        for filt in filters:
            for i in range(1,args.layer+1):
                with open(data_path+"/"+filt+"_"+str(i)+"_validation_"+str(seed)+".pickle","rb") as fopen:
                    valid_data.append(pkl.load(fopen))
        # test data
        test_data = []
        if args.model != "simplink":
            with open(data_path+"/feature_test_"+str(seed)+".pickle","rb") as fopen:
                test_data.append(pkl.load(fopen))
        for filt in filters:
            for i in range(1,args.layer+1):
                with open(data_path+"/"+filt+"_"+str(i)+"_test_"+str(seed)+".pickle","rb") as fopen:
                    test_data.append(pkl.load(fopen))
    elif args.model == "SGC" or args.model == "GCN_LC":
        # training data
        with open(data_path+"/adjacency_2_training_"+str(seed)+".pickle","rb") as fopen:
            train_data=[pkl.load(fopen)]
        # validation data
        with open(data_path+"/adjacency_2_validation_"+str(seed)+".pickle","rb") as fopen:
            valid_data=[pkl.load(fopen)]
        # test data
        with open(data_path+"/adjacency_2_test_"+str(seed)+".pickle","rb") as fopen:
            test_data=[pkl.load(fopen)]
    # elif args.model == "GPRGNN_LC3" or args.model == "GPRGNN_LC":
    #     # training data
    #     train_data = []
    #     with open(data_path+"/feature_training_"+str(seed)+".pickle","rb") as fopen:
    #         train_data.append(pkl.load(fopen))
    #     for i in range(1,args.layer+1):
    #         with open(data_path+"/adjacency_"+str(i)+"_training_"+str(seed)+".pickle","rb") as fopen:
    #             train_data.append(pkl.load(fopen))
    #     # validation data
    #     valid_data = []
    #     with open(data_path+"/feature_validation_"+str(seed)+".pickle","rb") as fopen:
    #         valid_data.append(pkl.load(fopen))
    #     for i in range(1,args.layer+1):
    #         with open(data_path+"/adjacency_"+str(i)+"_validation_"+str(seed)+".pickle","rb") as fopen:
    #             valid_data.append(pkl.load(fopen))
    #     # test data
    #     test_data = []
    #     with open(data_path+"/feature_test_"+str(seed)+".pickle","rb") as fopen:
    #         test_data.append(pkl.load(fopen))
    #     for i in range(1,args.layer+1):
    #         with open(data_path+"/adjacency_"+str(i)+"_test_"+str(seed)+".pickle","rb") as fopen:
    #             test_data.append(pkl.load(fopen))
                
    with open(data_path+"/labels_"+str(seed)+".pickle","rb") as fopen:
        split_idx, label = pkl.load(fopen)
    return train_data, valid_data, test_data, split_idx, label

def extract_few_labels(label_budget, train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_labels, smallest_class_threshold=3):
    too_small_class = set()
    if label_budget > 0:
        few_train = torch.tensor([], dtype=torch.int64)
        for l in range(num_labels):
            tmp_class_label = torch.flatten((train_labels==l).nonzero())
            # try:
            # print(len(tmp_class_label))
            if len(tmp_class_label) < smallest_class_threshold or len(tmp_class_label) < label_budget: ## if a class is smaller than a user-specified threshold, then it will be removed.
                too_small_class.add(l)
            else:
                tmp_labels = torch.tensor(np.random.choice(tmp_class_label, label_budget, replace=False), dtype=torch.int64)
                # print(tmp_labels)
                few_train = torch.concat([few_train,tmp_labels], axis=0)

        train_labels = train_labels[few_train]
        for i in range(len(train_data)):
            train_data[i] = train_data[i][few_train]#.to(device)
    # print(few_train)
    ## re-ordering classes
    class_order_dic = dict()
    all_class = set(range(num_labels))
    remaining_class = all_class-too_small_class
    for ind, l in enumerate(sorted(list(remaining_class))):
        class_order_dic[l] = ind
    num_few_labels = len(remaining_class)
    print(too_small_class)
    print("number of classes:", num_few_labels)
        
    # tmp_train_labels.append(original_train[few_train].reshape(-1).long())

    valid_wo_small = torch.tensor([], dtype=torch.int64)
    test_wo_small = torch.tensor([], dtype=torch.int64)
    for l in range(len(all_class)):
        if l in too_small_class:
            continue
        else:
            tmp_class_val = torch.flatten((valid_labels==l).nonzero())
            valid_wo_small = torch.concat([valid_wo_small,tmp_class_val], axis=0)
            tmp_class_test = torch.flatten((test_labels==l).nonzero())
            test_wo_small = torch.concat([test_wo_small,tmp_class_test], axis=0)
            
    for i in range(len(valid_data)):
        valid_data[i] = valid_data[i][valid_wo_small]
    for i in range(len(test_data)):
        test_data[i] = test_data[i][test_wo_small]

    def re_order(in_labels, class_order_dic):
        in_labels = in_labels.numpy()
        for i in range(len(in_labels)):
            in_labels[i] = class_order_dic[in_labels[i]]
        return torch.tensor(in_labels)
    train_labels = re_order(train_labels, class_order_dic)
    valid_labels = re_order(valid_labels[valid_wo_small], class_order_dic)
    test_labels = re_order(test_labels[test_wo_small], class_order_dic)
    
    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_few_labels


def load_precomp_augment(args, precomp_flag, data_path, sampling_ratio_list, few_train=None):
    import process_spmm_augment as process_spmm
    import pickle as pkl
    if precomp_flag == False:
        if args.model == "MLP":
            args.layer = 0
        args.filter = "adjacency"
        args.inductive = False
        for sampling_ratio in sampling_ratio_list:
            for ind in range(args.num_view):
                process_spmm.main(args, sampling_ratio=sampling_ratio, ind=ind)
        if args.model == "FSGNN":
            args.filter = "exact1hop"
            for sampling_ratio in sampling_ratio_list:
                for ind in range(args.num_view):
                    process_spmm.main(args, sampling_ratio=sampling_ratio, ind=ind)

        if args.dataset in ["flickr", "reddit"]:
            args.filter = "adjacency"
            args.inductive = True
            for sampling_ratio in sampling_ratio_list:
                for ind in range(args.num_view):
                    process_spmm.main(args, sampling_ratio=sampling_ratio, ind=ind)
            if args.model == "FSGNN":
                args.filter = "exact1hop"
                for sampling_ratio in sampling_ratio_list:
                    for ind in range(args.num_view):
                        process_spmm.main(args, sampling_ratio=sampling_ratio, ind=ind)
                
    if args.model == "FSGNN":
        filters = ["adjacency", "exact1hop"]
        layers = range(1, args.layer+1)
    elif args.model in ["JKnet_LC", "GPRGNN_LC", "GPRGNN_LC3"]:
        filters = ["adjacency"]
        layers = range(1, args.layer+1)
    else:
        filters = ["adjacency"]
        layers = [2]
    # training data
    def load_precomputed_data(data_tag, filters, layers, sampling_ratio_list):
        data_list = []
        num_view = args.num_view
        # with open(data_path+"/feature_"+data_tag+".pickle","rb") as fopen:
        #     feature_pickle = pickle.load(fopen)
        #     if data_tag == "training":
        #         feature_pickle = feature_pickle[few_train]
        # for sampling_ratio in sampling_ratio_list:
        #     for ind in range(args.num_view):
        #         if sampling_ratio == 1 and ind > 0:
        #             continue
        #         tmp_pickle.append(feature_pickle)
        # data_list.append(torch.concat(tmp_pickle))
        if data_tag != "training":
            sampling_ratio_list = [1]
            num_view = 1
        for sampling_ratio in sampling_ratio_list:
            for ind in range(num_view):
                if sampling_ratio != 1:
                    sampling_tag = "_" + str(sampling_ratio) + "_" + str(ind)
                else:
                    if ind > 0:
                        continue
                    sampling_tag = ""
                    
                tmp_pickle = []
                with open(data_path+"/feature_"+data_tag+".pickle","rb") as fopen:
                    feature_pickle = pkl.load(fopen)
                # if data_tag == "training":
                #     feature_pickle = feature_pickle[few_train]
                tmp_pickle.append(feature_pickle)
                
                for filt in filters:
                    for i in layers:
                        with open(data_path+"/"+filt+"_"+str(i)+sampling_tag+"_"+data_tag+".pickle","rb") as fopen:
                            # if data_tag == "training":
                            #     tmp = pickle.load(fopen)[few_train]
                            # else:
                            tmp = pkl.load(fopen)
                        tmp_pickle.append(tmp)    
                # data_list.append(torch.concat(tmp_pickle))
                data_list.append(tmp_pickle)
        return data_list

    train_data = load_precomputed_data("training", filters, layers, sampling_ratio_list)
    print(len(train_data[0]), train_data[0][0].shape)
    valid_data = load_precomputed_data("validation", filters, layers, sampling_ratio_list)[0]
    test_data = load_precomputed_data("test", filters, layers, sampling_ratio_list)[0]
    
    with open(data_path+"/labels.pickle","rb") as fopen:
        labels = pkl.load(fopen)
        
    return train_data, valid_data, test_data, labels

def extract_few_labels_augment(label_budget, train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_labels, smallest_class_threshold=3):
    too_small_class = set()
    if label_budget > 0:
        few_train = torch.tensor([], dtype=torch.int64)
        for l in range(num_labels):
            tmp_class_label = torch.flatten((train_labels==l).nonzero())
            # try:
            # print(max(tmp_class_label))
            if len(tmp_class_label) < smallest_class_threshold or len(tmp_class_label) < label_budget: ## if a class is smaller than a user-specified threshold, then it will be removed.
                too_small_class.add(l)
            else:
                tmp_labels = torch.tensor(np.random.choice(tmp_class_label, label_budget, replace=False), dtype=torch.int64)
                # print(tmp_labels)
                few_train = torch.concat([few_train,tmp_labels], axis=0)

        train_labels = train_labels[few_train]
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                train_data[i][j] = train_data[i][j][few_train]#.to(device)
    # print(few_train)
    ## re-ordering classes
    class_order_dic = dict()
    all_class = set(range(num_labels))
    remaining_class = all_class-too_small_class
    for ind, l in enumerate(sorted(list(remaining_class))):
        class_order_dic[l] = ind
    num_few_labels = len(remaining_class)
    print(too_small_class)
    print("number of classes:", num_few_labels)
        
    # tmp_train_labels.append(original_train[few_train].reshape(-1).long())

    valid_wo_small = torch.tensor([], dtype=torch.int64)
    test_wo_small = torch.tensor([], dtype=torch.int64)
    for l in range(len(all_class)):
        if l in too_small_class:
            continue
        else:
            tmp_class_val = torch.flatten((valid_labels==l).nonzero())
            valid_wo_small = torch.concat([valid_wo_small,tmp_class_val], axis=0)
            tmp_class_test = torch.flatten((test_labels==l).nonzero())
            test_wo_small = torch.concat([test_wo_small,tmp_class_test], axis=0)
            
    # print("max test:", max(test_wo_small))
    for i in range(len(valid_data)):
        valid_data[i] = valid_data[i][valid_wo_small]
    for i in range(len(test_data)):
        test_data[i] = test_data[i][test_wo_small]

    def re_order(in_labels, class_order_dic):
        in_labels = in_labels.numpy()
        for i in range(len(in_labels)):
            in_labels[i] = class_order_dic[in_labels[i]]
        return torch.tensor(in_labels)
    train_labels = re_order(train_labels, class_order_dic)
    valid_labels = re_order(valid_labels[valid_wo_small], class_order_dic)
    test_labels = re_order(test_labels[test_wo_small], class_order_dic)
    
    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels, num_few_labels


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#################################################################################
# Copy from DiGCN
# https://github.com/flyingtango/DiGCN
#################################################################################
def get_appr_directed_adj(alpha, edge_index, num_nodes, dtype, edge_weight=None):
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(edge_index.long(), edge_weight, fill_value, num_nodes)  
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes) 
    deg_inv = deg.pow(-1) 
    deg_inv[deg_inv == float("inf")] = 0
    p = deg_inv[row] * edge_weight 

    # personalized pagerank p
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    p_v = torch.zeros(torch.Size([num_nodes+1,num_nodes+1]))
    p_v[0:num_nodes,0:num_nodes] = (1-alpha) * p_dense
    p_v[num_nodes,0:num_nodes] = 1.0 / num_nodes
    p_v[0:num_nodes,num_nodes] = alpha
    p_v[num_nodes,num_nodes] = 0.0
    p_ppr = p_v 

    eig_value, left_vector = scipy.linalg.eig(p_ppr.numpy(),left=True,right=False)
    eig_value = torch.from_numpy(eig_value.real)
    left_vector = torch.from_numpy(left_vector.real)
    val, ind = eig_value.sort(descending=True)

    pi = left_vector[:,ind[0]] # choose the largest eig vector
    pi = pi[0:num_nodes]
    p_ppr = p_dense
    pi = pi/pi.sum()  # norm pi

    # Note that by scaling the vectors, even the sign can change. That"s why positive and negative elements might get flipped.
    assert len(pi[pi<0]) == 0

    pi_inv_sqrt = pi.pow(-0.5)
    pi_inv_sqrt[pi_inv_sqrt == float("inf")] = 0
    pi_inv_sqrt = pi_inv_sqrt.diag()
    pi_sqrt = pi.pow(0.5)
    pi_sqrt[pi_sqrt == float("inf")] = 0
    pi_sqrt = pi_sqrt.diag()

    # L_appr
    L = (torch.mm(torch.mm(pi_sqrt, p_ppr), pi_inv_sqrt) + torch.mm(torch.mm(pi_inv_sqrt, p_ppr.t()), pi_sqrt)) / 2.0

    # make nan to 0
    L[torch.isnan(L)] = 0

    # transfer dense L to sparse
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values

    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

def get_second_directed_adj(edge_index, num_nodes, dtype, edge_weight=None):
    from torch_geometric.utils import add_remaining_self_loops, add_self_loops, remove_self_loops
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
    fill_value = 1
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv[deg_inv == float("inf")] = 0
    p = deg_inv[row] * edge_weight 
    p_dense = torch.sparse.FloatTensor(edge_index, p, torch.Size([num_nodes,num_nodes])).to_dense()
    
    L_in = torch.mm(p_dense.t(), p_dense)
    L_out = torch.mm(p_dense, p_dense.t())
    
    L_in_hat = L_in
    L_out_hat = L_out

    L_in_hat[L_out == 0] = 0
    L_out_hat[L_in == 0] = 0

    # L^{(2)}
    L = (L_in_hat + L_out_hat) / 2.0

    L[torch.isnan(L)] = 0
    L_indices = torch.nonzero(L,as_tuple=False).t()
    L_values = L[L_indices[0], L_indices[1]]
    edge_index = L_indices
    edge_weight = L_values
    
    # row normalization
    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

#################################################################################
# Copy from MagNet repository
# https://github.com/matthew-hirn/magnet/blob/cb730c7e87b6f38740480c656d441d6ac5369cd3/src/utils/preprocess.py#L229
#################################################################################
def F_in_out(edge_index, size, edge_weight=None):
    if edge_weight is not None:
        a = sp.coo_matrix((edge_weight, edge_index), shape=(size, size)).tocsc()
    else:
        a = sp.coo_matrix((np.ones(len(edge_index[0])), edge_index), shape=(size, size)).tocsc()
    
    out_degree = np.array(a.sum(axis=0))[0]
    out_degree[out_degree == 0] = 1

    in_degree = np.array(a.sum(axis=1))[:, 0]
    in_degree[in_degree == 0] = 1
    '''
    # can be more efficient
    a = np.zeros((size, size), dtype=np.uint8)
    a[edge_index[0], edge_index[1]] = 1
    out_degree = np.sum(a, axis = 1)
    out_degree[out_degree == 0] = 1
    
    in_degree = np.sum(a, axis = 0)
    in_degree[in_degree == 0] = 1
    '''
    # sparse implementation
    a = sp.csr_matrix(a)
    # A_in = sp.csr_matrix(np.zeros((size, size)))
    # A_out = sp.csr_matrix(np.zeros((size, size)))
    A_in = sp.coo_matrix(([],[[],[]]), shape=(size,size)).tocsr()
    A_out = sp.coo_matrix(([],[[],[]]), shape=(size,size)).tocsr()
    for k in range(size):
        A_in += np.dot(a[k, :].T, a[k, :])/out_degree[k]
        A_out += np.dot(a[:,k], a[:,k].T)/in_degree[k]

    A_in = A_in.tocoo()
    A_out = A_out.tocoo()

    edge_in  = torch.from_numpy(np.vstack((A_in.row,  A_in.col))).long()
    edge_out = torch.from_numpy(np.vstack((A_out.row, A_out.col))).long()
    
    in_weight  = torch.from_numpy(A_in.data).float()
    out_weight = torch.from_numpy(A_out.data).float()
    return to_undirected(edge_index), edge_in, in_weight, edge_out, out_weight