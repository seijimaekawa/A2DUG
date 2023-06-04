import math
import numpy as np
import random
from typing import Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Linear, ModuleList, Sequential, BatchNorm1d, ReLU
from torch.nn import Parameter

import torch_geometric.transforms as T
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.nn import GATConv, GCNConv, ChebConv, SAGEConv, SGConv, GMMConv, GraphConv
from torch_geometric.nn.models import GraphSAGE, GIN, GAT
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops

from layers import GraphConvolution, SparseNGCNLayer, ListModule, DenseNGCNLayer, SparseLinear, MLP, GraphConvolution_acm
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_add

   
class simplink(torch.nn.Module):
    def __init__(self, args, nlayers):
        super(simplink, self).__init__()
        self.num_nodes = args.num_nodes
        self.in_channels = args.num_features
        self.out_channels = args.C
        self.num_edge_layers = args.num_edge_layers
        self.agg = args.agg
        self.wo_agg = args.wo_agg
        self.wo_mlp = args.wo_mlp
        self.wo_undirected = args.wo_undirected
        self.wo_directed = args.wo_directed
        self.wo_adj = args.wo_adj
        self.wo_att = args.wo_att
        self.wo_transpose = args.wo_transpose
        self.nlayers = nlayers
        self.normalization = False
        
        self.total_layers = 0
        nAGG = 0
        if not self.wo_agg:
            if not self.wo_undirected:
                self.mlp_agg = MLP(self.in_channels*nlayers*2, args.hidden, args.hidden, args.num_agg_layers, dropout=0)
                nAGG += 1
            if not self.wo_directed:
                self.mlp_agg_di = MLP(self.in_channels*nlayers*2, args.hidden, args.hidden, args.num_agg_layers, dropout=0)
                nAGG += 1
                if not self.wo_transpose:
                    self.mlp_agg_di_t = MLP(self.in_channels*nlayers*2, args.hidden, args.hidden, args.num_agg_layers, dropout=0)
                    nAGG += 1
        # self.wt1 = nn.ModuleList([nn.Linear(self.in_channels,int(args.hidden)) for _ in range(self.total_layers)])
        # self.wt1 = nn.ModuleList([MLP(self.in_channels, args.hidden, args.hidden, args.num_agg_layers, dropout=0) for _ in range(self.nlayers)])
        
        nMLP=0
        if not self.wo_mlp:
            self.mlpX = MLP(self.in_channels, args.hidden, args.hidden, args.num_node_layers, dropout=0)
            nMLP += 1
        nADJ=0
        if not self.wo_adj:
            if not self.wo_undirected:
                self.mlpA = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
                nADJ += 1
            if not self.wo_directed:
                self.mlpA_di = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
                nADJ += 1
                if not self.wo_transpose:
                    self.mlpA_di_t = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
                    nADJ += 1
        # self.W = nn.Linear(3*args.hidden, args.hidden) # for MLP_X, MLP_A, GNN
        if self.agg == 'concat':
            self.mlp_final = MLP(args.hidden*(nAGG+nMLP+nADJ), args.hidden, self.out_channels, args.final_layers, dropout=args.dropout)
        elif self.agg == 'sum':
            self.mlp_final = MLP(args.hidden, args.hidden, self.out_channels, args.final_layers, dropout=args.dropout)
        self.A = None
        
        # TEMP = np.ones(7) / 7
        # self.att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))        
        self.sm = nn.Softmax(dim=0)
        self.sm_x = nn.Softmax(dim=1)
        
        # self.inner_activation = inner_activation
        # self.inner_dropout = inner_dropout
        if not self.wo_att:
            TEMP = np.ones(nAGG+nMLP+nADJ)
            self.att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))
            # self.adj_att = nn.Parameter(torch.tensor(adj_TEMP, dtype=torch.float32))
            
            # if not self.wo_undirected:
            #     TEMP = np.ones(nlayers*2)
            #     self.agg_att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))
            # if not self.wo_directed:
            #     TEMP = np.ones(nlayers*2)
            #     self.agg_di_att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))
            #     TEMP = np.ones(nlayers*2)
            #     self.agg_di_t_att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))
            

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	
        
    def encoder(self, x, A, A_di, A_di_t, list_adj, device, st=0, end=0, edge_weight=None):
        x = x[st:end].to(device)
        A_di = A_di[st:end].to_torch_sparse_coo_tensor().to(device)
        if not self.wo_transpose:
            A_di_t = A_di_t[st:end].to_torch_sparse_coo_tensor().to(device)
        A = A[st:end].to_torch_sparse_coo_tensor().to(device)

        out = []
        
        if not self.wo_mlp:
            xX = self.mlpX(x, input_tensor=True)
            out.append(xX)
        if not self.wo_adj:
            if not self.wo_undirected:
                xA = self.mlpA(A, input_tensor=True)
                out.append(xA)
            if not self.wo_directed:
                xA_di = self.mlpA_di(A_di, input_tensor=True)
                out.append(xA_di)
                if not self.wo_transpose:
                    xA_di_t = self.mlpA_di_t(A_di_t, input_tensor=True)
                    out.append(xA_di_t)
        
        # out = [xX,xA,xA_di,xA_di_t]
        if not self.wo_agg:
            if not self.wo_undirected:
                # und_out = []
                # mask = self.sm(self.agg_att)
                mat = []
                for i in range(self.nlayers*2): # directedのみのケースで修正が必要
                    mat.append(list_adj[i][st:end].to(device))
                mat = torch.concat(mat, axis=1)
                out.append(self.mlp_agg(mat, input_tensor=True))
                    # tmp = self.sm_x(self.wt1[i](mat))
                    # tmp = torch.mul(mask[i], tmp)
                    # und_out.append(tmp)
            if not self.wo_directed:
                # dire_out = []
                # mask = self.sm(self.agg_di_att)
                mat = []
                for i in range(self.nlayers*2):
                    mat.append(list_adj[self.nlayers*2+i][st:end].to(device))
                mat = torch.concat(mat, axis=1)
                out.append(self.mlp_agg_di(mat, input_tensor=True))

                if not self.wo_transpose:
                    mat = []
                    for i in range(self.nlayers*2):
                        mat.append(list_adj[self.nlayers*4+i][st:end].to(device))
                    mat = torch.concat(mat, axis=1)
                    out.append(self.mlp_agg_di_t(mat, input_tensor=True))
            
        if not self.wo_att:
            mask = self.sm(self.att)
            for i in range(len(out)):
                if self.normalization:
                    out[i] = self.sm_x(out[i])
                out[i] = torch.mul(mask[i], out[i])            
        if self.agg == 'concat':
            agg_out = torch.concat(out, axis=1)
        elif self.agg == 'sum':
            agg_out = 0
            for i in range(len(out)):
                agg_out += out[i]
        return agg_out
    
    def forward(self, x, A, A_di, A_di_t, list_adj, device, st=0, end=0, edge_weight=None):
        x = self.encoder(x, A, A_di, A_di_t, list_adj, device, st=st, end=end)
        x = F.relu(x)
        x = self.mlp_final(x, input_tensor=True)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)

# from https://github.com/Tiiiger/SGC
class SGC(nn.Module):
    def __init__(self,nfeat,nclass,nhidden=None):
        super(SGC, self).__init__()
        self.W = nn.Linear(nfeat, nclass)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.W.reset_parameters()
    
    def encoder(self,list_adj,device,st=0,end=0):
        # for ind, mat in enumerate(list_adj):
            # mat = mat.to(device)
            # mat = mat[st:end,:].to(device)
        mat = list_adj[0][st:end,:].to(device)
        x = self.W(mat)
        return x

    def forward(self,list_adj,device,st=0,end=0):
        x = self.encoder(list_adj,device,st,end)
        return F.log_softmax(x,dim=1), F.softmax(x, dim=1)
    
# from https://github.com/sunilkmaurya/FSGNN
class FSGNN_Large(nn.Module):
    def __init__(self,nfeat,nlayers,nhidden,nclass,dp1,dp2,layer_norm=True):
        super(FSGNN_Large,self).__init__()
        self.wt1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.fc2 = nn.Linear(nhidden*nlayers,nhidden)
        self.fc3 = nn.Linear(nhidden,nclass)
        self.dropout1 = dp1 
        self.dropout2 = dp2 
        self.act_fn = nn.ReLU()
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.layer_norm = layer_norm
        self.reset_parameters()

    def reset_parameters(self):
        for ind in range(len(self.wt1)):
            self.wt1[ind].reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        
    def encoder(self,list_adj,device,st=0,end=0):
        mask = self.sm(self.att)
        mask = torch.mul(len(list_adj),mask)
        list_out = list()
        for ind, mat in enumerate(list_adj):
            # mat = mat.to(device)
            mat = mat[st:end,:].to(device)
            tmp_out = self.wt1[ind](mat)
            if self.layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)
            list_out.append(tmp_out)
        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout1,training=self.training)
        out = self.fc2(out)
        # out = self.act_fn(out)
        # out = F.dropout(out,self.dropout2,training=self.training)
        # out = self.fc3(out)
        return out
        
    def forward(self,list_adj,device,st=0,end=0):
        out = self.encoder(list_adj, device, st, end)
        out = self.act_fn(out)
        out = F.dropout(out,self.dropout2,training=self.training)
        out = self.fc3(out)
        return F.log_softmax(out, dim=1), F.softmax(out, dim=1)

class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP, dtype=torch.float32))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            # x = torch.spmm(edge_index, x)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.C)
        self.prop1 = GPR_prop(10, args.alpha, "PPR", None)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()
        

    def forward(self, features, edge_index):
        x = features

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            # return F.log_softmax(x, dim=1)
            return F.log_softmax(x,dim=1), F.softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            # return F.log_softmax(x, dim=1)
            return F.log_softmax(x,dim=1), F.softmax(x, dim=1)

## from https://github.com/tkipf/pygcn
class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(args.num_features, args.hidden)
        self.gc2 = GraphConvolution(args.hidden, args.C)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        
    def encoder(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.spmm(adj, x)
        return x

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x,dim=1), F.softmax(x, dim=1)

class GCN_Net(torch.nn.Module):
    def __init__(self, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.C)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, features, edge_index):
        torch.cuda.empty_cache()
        edge_index = edge_index
        x = features
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x,dim=1), F.softmax(x, dim=1)

class GCN_Net2(torch.nn.Module):
    # 元々GPRGNNで書いてあったGCNをgraphSAINT用に改変
    def __init__(self, args):
        super(GCN_Net2, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.C)
        self.dropout = args.dropout
        self.edge_index = args.edge_index

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, features, edge_index, edge_weight=None):
        x = features
        edge_index = edge_index
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        edge_index.to('cpu')
        return F.log_softmax(x, dim=1)

class JKnet(GCN):
    def __init__(self, args):
        super().__init__(
            in_channels=args.num_features,
            out_channels=args.C,
            hidden_channels=int(args.hidden),
            num_layers=3,
            dropout=args.dropout,
            act=ReLU(inplace=True),
            norm=None,
            jk=args.JK
        )
        self.edge_index = args.edge_index

    def forward(self, features, edge_index):
        edge_index = edge_index
        x = features
        x_final = super().forward(x, edge_index)
        return F.log_softmax(x_final, dim=1)

    def set_aggr(self, aggr):
        for conv in self.convs:
            conv.aggr = aggr

class JKnet3(torch.nn.Module):
    def __init__(self, args):
        super(JKnet3, self).__init__()
        self.pooling = args.JK
        self.nlayers = 3
        self.conv1 = GraphConvolution(args.num_features, args.hidden)
        self.conv = torch.nn.ModuleList([GraphConvolution(args.hidden, args.hidden) for _ in range(self.nlayers-1)])
        if self.pooling == "cat":
            self.W_final = torch.nn.Linear(args.hidden*self.nlayers, args.C)
        # elif self.pooling == "max":
        #     self.W_final = torch.nn.Linear(args.hidden, args.C)
        self.act_fn = torch.nn.ReLU()
        self.dropout = args.dropout
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for ind in range(len(self.conv)):
            self.conv[ind].reset_parameters()
        if self.pooling == "cat":
            self.W_final.reset_parameters()

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x_list = [x]
        for i in range(2):
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv[i](x, adj)
            x_list.append(x)
        if self.pooling == "cat":
            x = torch.concat(x_list,dim=1)
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.W_final(x)
        elif self.pooling == "max":
            x = torch.stack(x_list).max(dim=0).values
        # return F.log_softmax(x,dim=1)
        return F.log_softmax(x,dim=1), F.softmax(x, dim=1)

class ChebNet(torch.nn.Module):
    def __init__(self, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(args.num_features, args.hidden, K=2)
        self.conv2 = ChebConv(args.hidden, args.C, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, self.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            args.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            args.C,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        # x, edge_index = data.x, self.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x,dim=1), F.softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.C)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, self.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x,dim=1), F.softmax(x, dim=1)


class GCN_JKNet(torch.nn.Module):
    def __init__(self, args):
        in_channels = args.num_features
        out_channels = args.C

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = torch.nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=16,
                                   num_layers=4
                                   )

    def forward(self, data):
        x, edge_index = data.x, self.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)
    
class LINKX(torch.nn.Module):
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """
    def __init__(self, args, inner_activation=False, inner_dropout=False):
        super(LINKX, self).__init__()
        self.num_nodes = args.num_nodes
        self.in_channels = args.num_features
        self.out_channels = args.C
        self.num_edge_layers = args.num_edge_layers
        
        self.mlpA = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
        self.mlpX = MLP(self.in_channels, args.hidden, args.hidden, args.num_node_layers, dropout=0)
        self.W = nn.Linear(2*args.hidden, args.hidden)
        self.mlp_final = MLP(args.hidden, args.hidden, self.out_channels, args.layers, dropout=args.dropout)
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	
        
    def encoder(self, x, A, device, st=0, end=0, edge_weight=None):
        x = x[st:end].to(device)
        A = A[st:end].to_torch_sparse_coo_tensor().to(device)
        xA = self.mlpA(A, input_tensor=True)
        # xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        xX = self.mlpX(x, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = x + xA + xX
        return x
    
    # def forward(self, data):	
    def forward(self, x, A, device, st=0, end=0, edge_weight=None):
        x = self.encoder(x, A, device, st, end)
        x = F.relu(x)
        x = self.mlp_final(x, input_tensor=True)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)
        
class LINK(torch.nn.Module):
    def __init__(self, args):
        super(LINK, self).__init__()
        self.num_nodes = args.num_nodes
        self.in_channels = args.num_features
        self.out_channels = args.C
        self.num_edge_layers = args.num_edge_layers
        
        self.mlpA = MLP(self.num_nodes, args.hidden, self.out_channels, args.num_edge_layers, dropout=0)

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	
        
    def encoder(self, features, A, device, st=0, end=0, edge_weight=None):
        A = A[st:end].to_torch_sparse_coo_tensor().to(device)
        x = self.mlpA(A, input_tensor=True)
        return x
    
    # def forward(self, data):	
    def forward(self, features, A, device, st=0, end=0, edge_weight=None):
        x = self.encoder(features, A, device, st, end)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)
    
class MLPNet(torch.nn.Module):
    def __init__(self, args):
        super(MLPNet, self).__init__()

        self.lin1 = Linear(args.num_features, int(args.hidden))
        self.lin2 = Linear(int(args.hidden), args.C)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)

class MLP_minibatch(torch.nn.Module): # mini-batch training
    def __init__(self,nfeat,nclass,nhidden,nlayer,dropout):
        super(MLP_minibatch, self).__init__()
        self.nlayer = nlayer
        self.lin1 = Linear(nfeat, nhidden)
        if nlayer == 3:
            self.lin2 = Linear(nhidden, nhidden)
        self.lin3 = Linear(nhidden, nclass)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
    
    def encoder(self,list_adj,device,st=0,end=0):
        x = list_adj[0][st:end].to(device)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.nlayer == 3:
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
        return x

    def forward(self,list_adj,device,st=0,end=0):
        x = self.encoder(list_adj,device,st,end)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)
    
class LINKGNN(torch.nn.Module):
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
    def __init__(self, args, gnn, gnn_di=None, gnn_di_t=None):
        super(LINKGNN, self).__init__()
        self.num_nodes = args.num_nodes
        self.in_channels = args.num_features
        self.out_channels = args.C
        self.num_edge_layers = args.num_edge_layers
        self.gnn = gnn
        self.gnn_di = gnn_di
        self.gnn_di_t = gnn_di_t
        
        self.mlpA = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
        self.mlpA_t = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
        self.mlpX = MLP(self.in_channels, args.hidden, args.hidden, args.num_node_layers, dropout=0)
        # self.W = nn.Linear(3*args.hidden, args.hidden) # for MLP_X, MLP_A, GNN
        self.mlp_final = MLP(args.hidden, args.hidden, self.out_channels, args.linkx_layers, dropout=args.linkx_dropout)
        self.A = None
        
        TEMP = np.ones(7) / 7
        self.att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))        
        self.sm = nn.Softmax(dim=0)
        self.sm_x = nn.Softmax(dim=1)
        
        # self.inner_activation = inner_activation
        # self.inner_dropout = inner_dropout
        self.inner_activation = False
        self.inner_dropout = False

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	
        
    def encoder(self, x, A, A_di, A_di_t, list_adj, device, st=0, end=0, edge_weight=None):
        x = x[st:end].to(device)
        A = A[st:end].to_torch_sparse_coo_tensor().to(device)
        A_di = A_di[st:end].to_torch_sparse_coo_tensor().to(device)
        A_di_t = A_di_t[st:end].to_torch_sparse_coo_tensor().to(device)

        xA = self.sm_x(self.mlpA(A, input_tensor=True))
        xA_di = self.sm_x(self.mlpA_di(A_di, input_tensor=True))
        xA_di_t = self.sm_x(self.mlpA_di_t(A_di_t, input_tensor=True))
        xX = self.sm_x(self.mlpX(x, input_tensor=True))
        
        xGNN = self.sm_x(self.gnn.encoder(list_adj[0],device,st=st,end=end))
        xGNN_di = self.sm_x(self.gnn_di.encoder(list_adj[1],device,st=st,end=end))
        xGNN_di_t = self.sm_x(self.gnn_di_t.encoder(list_adj[2],device,st=st,end=end))
        
        mask = self.sm(self.att)
        x = mask[0]*xA + mask[1]*xA_di + mask[2]*xA_di_t + mask[3]*xX + mask[4]*xGNN + mask[5]*xGNN_di + mask[6]*xGNN_di_t
        return x
    
    # def forward(self, data):	
    def forward(self, x, A, A_di, A_di_t, list_adj, device, st=0, end=0, edge_weight=None):
        x = self.encoder(x, A, A_di, A_di_t, list_adj, device, st=st, end=end)
        x = F.relu(x)
        x = self.mlp_final(x, input_tensor=True)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)
        # return x

class simp(torch.nn.Module):
    # def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
    def __init__(self, args, nlayers):
        super(simp, self).__init__()
        self.num_nodes = args.num_nodes
        self.in_channels = args.num_features
        self.out_channels = args.C
        self.num_edge_layers = args.num_edge_layers
        self.agg = args.agg
        self.wo_mlp = args.wo_mlp
        self.wo_undirected = args.wo_undirected
        self.wo_directed = args.wo_directed
        self.wo_adj = args.wo_adj
        
        self.nlayers = 0
        if not self.wo_undirected:
            self.nlayers += nlayers*2
        if not self.wo_directed:
            self.nlayers += nlayers*4
        self.wt1 = nn.ModuleList([nn.Linear(self.in_channels,int(args.hidden)) for _ in range(self.nlayers)])
        # self.wt1 = nn.ModuleList([MLP(self.in_channels, args.hidden, args.hidden, args.num_agg_layers, dropout=0) for _ in range(self.nlayers)])
        
        nMLP=0
        if not self.wo_mlp:
            self.mlpX = MLP(self.in_channels, args.hidden, args.hidden, args.num_node_layers, dropout=0)
            nMLP += 1
        if not self.wo_adj:
            if not self.wo_undirected:
                self.mlpA = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
                nMLP += 1
            if not self.wo_directed:
                self.mlpA_di = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
                self.mlpA_di_t = MLP(self.num_nodes, args.hidden, args.hidden, args.num_edge_layers, dropout=0)
                nMLP += 2
        # self.W = nn.Linear(3*args.hidden, args.hidden) # for MLP_X, MLP_A, GNN
        if self.agg == 'concat':
            self.mlp_final = MLP(args.hidden*(self.nlayers+nMLP), args.hidden, self.out_channels, args.final_layers, dropout=args.dropout)
        elif self.agg == 'sum':
            self.mlp_final = MLP(args.hidden, args.hidden, self.out_channels, args.final_layers, dropout=args.dropout)
        self.A = None
        
        # TEMP = np.ones(7) / 7
        # self.att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))        
        self.sm = nn.Softmax(dim=0)
        self.sm_x = nn.Softmax(dim=1)
        
        # self.inner_activation = inner_activation
        # self.inner_dropout = inner_dropout
        self.attention = True
        if self.attention:
            num_att = self.nlayers + nMLP
            TEMP = np.ones(num_att)
            self.att = nn.Parameter(torch.tensor(TEMP, dtype=torch.float32))
            

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	
        
    def encoder(self, x, A, A_di, A_di_t, list_adj, device, st=0, end=0, edge_weight=None):
        x = x[st:end].to(device)
        A_di = A_di[st:end].to_torch_sparse_coo_tensor().to(device)
        A_di_t = A_di_t[st:end].to_torch_sparse_coo_tensor().to(device)
        A = A[st:end].to_torch_sparse_coo_tensor().to(device)

        out = []
        
        if not self.wo_mlp:
            xX = self.sm_x(self.mlpX(x, input_tensor=True))
            out.append(xX)
        if not self.wo_adj:
            if not self.wo_undirected:
                xA = self.sm_x(self.mlpA(A, input_tensor=True))
                out.append(xA)
            if not self.wo_directed:
                xA_di = self.sm_x(self.mlpA_di(A_di, input_tensor=True))
                out.append(xA_di)
                xA_di_t = self.sm_x(self.mlpA_di_t(A_di_t, input_tensor=True))
                out.append(xA_di_t)
        
        # out = [xX,xA,xA_di,xA_di_t]
        for i in range(len(self.wt1)): # directedのみのケースで修正が必要
            mat = list_adj[i][st:end].to(device)
            out.append(self.sm_x(self.wt1[i](mat)))
        if self.attention:
            mask = self.sm(self.att)
            for i in range(len(out)):
                out[i] = torch.mul(mask[i], out[i])            
        if self.agg == 'concat':
            agg_out = torch.concat(out, axis=1)
        elif self.agg == 'sum':
            agg_out = 0
            for i in range(len(out)):
                agg_out += out[i]
        return agg_out
    
    def forward(self, x, A, A_di, A_di_t, list_adj, device, st=0, end=0, edge_weight=None):
        x = self.encoder(x, A, A_di, A_di_t, list_adj, device, st=st, end=end)
        x = F.relu(x)
        x = self.mlp_final(x, input_tensor=True)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)
    

def process(mul_L_real, mul_L_imag, weight, X_real, X_imag):
    data = torch.spmm(mul_L_real, X_real)
    real = torch.matmul(data, weight) 
    data = -1.0*torch.spmm(mul_L_imag, X_imag)
    real += torch.matmul(data, weight) 
    
    data = torch.spmm(mul_L_imag, X_real)
    imag = torch.matmul(data, weight)
    data = torch.spmm(mul_L_real, X_imag)
    imag += torch.matmul(data, weight)
    return torch.stack([real, imag])

##### from https://github.com/matthew-hirn/magnet/blob/main/src/node_classification.py #####
############################################################################################

class ChebConv(nn.Module):
    """
    The MagNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    :param L_norm_real, L_norm_imag: normalized laplacian of real and imag
    """
    def __init__(self, in_c, out_c, K,  L_norm_real, L_norm_imag, bias=True):
        super(ChebConv, self).__init__()

        L_norm_real, L_norm_imag = L_norm_real, L_norm_imag

        # list of K sparsetensors, each is N by N
        self.mul_L_real = L_norm_real   # [K, N, N]
        self.mul_L_imag = L_norm_imag   # [K, N, N]

        self.weight = nn.Parameter(torch.Tensor(K + 1, in_c, out_c))  # [K+1, 1, in_c, out_c]

        stdv = 1. / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, data):
        """
        :param inputs: the input data, real [B, N, C], img [B, N, C]
        :param L_norm_real, L_norm_imag: the laplace, [N, N], [N,N]
        """
        X_real, X_imag = data[0], data[1]

        real = 0.0
        imag = 0.0

        future = []
        for i in range(len(self.mul_L_real)): # [K, B, N, D]
            future.append(torch.jit.fork(process, 
                            self.mul_L_real[i], self.mul_L_imag[i], 
                            self.weight[i], X_real, X_imag))
        result = []
        for i in range(len(self.mul_L_real)):
            result.append(torch.jit.wait(future[i]))
        result = torch.sum(torch.stack(result), dim=0)

        real = result[0]
        imag = result[1]
        return real + self.bias, imag + self.bias

class complex_relu_layer(nn.Module):
    def __init__(self, ):
        super(complex_relu_layer, self).__init__()
    
    def complex_relu(self, real, img):
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real, img=None):
        # for torch nn sequential usage
        # in this case, x_real is a tuple of (real, img)
        if img == None:
            img = real[1]
            real = real[0]

        real, img = self.complex_relu(real, img)
        return real, img

class Magnet(nn.Module): #ChebNet(nn.Module):
    # def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim=2, activation=False, layer=2, dropout=False):
    def __init__(self, args):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(Magnet, self).__init__()
        in_c = args.num_features
        L_norm_real = args.L_real
        L_norm_imag = args.L_img
        num_filter = args.num_filter
        K = args.K
        label_dim = args.C
        activation = True
        layer = args.layer
        dropout = args.dropout
        
        
        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation:
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())

        self.Chebs = torch.nn.Sequential(*chebs)

        last_dim = 2  
        self.Conv = nn.Conv1d(num_filter*last_dim, label_dim, kernel_size=1)        
        self.dropout = dropout

    def forward(self, real, imag):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real, imag), dim = -1)
        
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)
        # return x

class ChebNet_Edge(nn.Module):
    def __init__(self, in_c, L_norm_real, L_norm_imag, num_filter=2, K=2, label_dim = 2, activation = False, layer = 2, dropout = False):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param K: for cheb series
        :param L_norm_real, L_norm_imag: normalized laplacian
        """
        super(ChebNet_Edge, self).__init__()
        
        chebs = [ChebConv(in_c=in_c, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag)]
        if activation and (layer != 1):
            chebs.append(complex_relu_layer())

        for i in range(1, layer):
            chebs.append(ChebConv(in_c=num_filter, out_c=num_filter, K=K, L_norm_real=L_norm_real, L_norm_imag=L_norm_imag))
            if activation:
                chebs.append(complex_relu_layer())
        self.Chebs = torch.nn.Sequential(*chebs)
        
        last_dim = 2
        self.linear = nn.Linear(num_filter*last_dim*2, label_dim)     
        self.dropout = dropout

    def forward(self, real, imag, index):
        real, imag = self.Chebs((real, imag))
        x = torch.cat((real[index[:,0]], real[index[:,1]], imag[index[:,0]], imag[index[:,1]]), dim = -1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class DIGCNConv(MessagePassing):
    r"""The graph convolutional operator takes from Pytorch Geometric.
    The spectral operation is the same with Kipf's GCN.
    DiGCN preprocesses the adjacency matrix and does not require a norm operation during the convolution operation.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the adj matrix on first execution, and will use the
            cached version for further executions.
            Please note that, all the normalized adj matrices (including undirected)
            are calculated in the dataset preprocessing to reduce time comsume.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, improved=False, cached=True,
                 bias=True, **kwargs):
        super(DIGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
    
    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if edge_weight is None:
                raise RuntimeError(
                    'Normalized adj matrix cannot be None. Please '
                    'obtain the adj matrix in preprocessing.')
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

    
class DiModel(torch.nn.Module):
    # def __init__(self, input_dim, out_dim, filter_num, dropout = False, layer=2):
    def __init__(self, args):
        super(DiModel, self).__init__()
        input_dim = args.num_features
        out_dim = args.C
        filter_num = args.num_filter
        dropout = args.dropout
        layer = args.layer
        
        self.conv1 = DIGCNConv(input_dim, filter_num)
        self.conv2 = DIGCNConv(filter_num, filter_num)
        
        # self.layer = layer
        self.layer = layer
        if self.layer == 3:
            self.conv3 = DIGCNConv(filter_num, filter_num)

        self.Conv = nn.Conv1d(filter_num, out_dim, kernel_size=1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        if self.layer==3:
            x = F.relu(self.conv3(x, edge_index, edge_weight))

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)

class DiGCNet(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden, dropout = False):
        super(DiGCNet, self).__init__()
        self.conv1 = DIGCNConv(input_dim, hidden)
        self.conv2 = DIGCNConv(hidden, hidden)
        self.linear = nn.Linear(hidden*2, out_dim)     
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)

class InceptionBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(InceptionBlock, self).__init__()
        self.ln = Linear(in_dim, out_dim)
        self.conv1 = DIGCNConv(in_dim, out_dim)
        self.conv2 = DIGCNConv(in_dim, out_dim)
    def reset_parameters(self):
        self.ln.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    def forward(self, x, edge_index, edge_weight, edge_index2, edge_weight2):
        x0 = self.ln(x)
        x1 = self.conv1(x, edge_index, edge_weight)
        x2 = self.conv2(x, edge_index2, edge_weight2)
        return x0, x1, x2

class DiGCN_IB(torch.nn.Module):
    # def __init__(self, num_features, hidden, num_classes, dropout=0.5, layer = 2):
    def __init__(self, args):
        super(DiGCN_IB, self).__init__()
        num_features = args.num_features
        hidden = args.num_filter
        num_classes = args.C
        dropout = args.dropout
        layer = args.layer
        
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, hidden)
        self._dropout = dropout
        self.Conv = nn.Conv1d(hidden, num_classes, kernel_size=1)

        self.layer = layer
        if layer == 3:
            self.ib3 = InceptionBlock(hidden, hidden)

    def forward(self, features, edge_index_tuple, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        if self.layer == 3:
            x0,x1,x2 = self.ib3(x, edge_index, edge_weight, edge_index2, edge_weight2)
            x = x0+x1+x2

        x = F.dropout(x, p=self._dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()
        # return F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)

class DiGCNet_IB(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden, dropout = False):
        super(DiGCNet_IB, self).__init__()
        self.ib1 = InceptionBlock(num_features, hidden)
        self.ib2 = InceptionBlock(hidden, hidden)
        self.linear = nn.Linear(hidden*2, num_classes)   
        self.dropout = dropout

    def forward(self, features, edge_index_tuple, index, edge_weight_tuple):
        x = features
        edge_index, edge_index2 = edge_index_tuple
        edge_weight, edge_weight2 = edge_weight_tuple
        x0,x1,x2 = self.ib1(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        x0,x1,x2 = self.ib2(x, edge_index, edge_weight, edge_index2, edge_weight2)
        x = x0+x1+x2
        
        x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

class ACMGCN(nn.Module):
    # def __init__(self, nfeat, nhid, nclass, dropout, model_type, nlayers=1, variant=False):
    def __init__(self, args):
        super(ACMGCN, self).__init__()
        nfeat = args.num_features
        nhid = args.hidden
        nclass = args.C
        dropout = args.dropout
        model_type = "acmgcn"
        nlayers = args.layer
        variant = False
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.nlayers, = model_type, nlayers
        if self.model_type == 'mlp':
            self.gcns.append(GraphConvolution_acm(
                nfeat, nhid, model_type=model_type))
            self.gcns.append(GraphConvolution_acm(
                nhid, nclass, model_type=model_type, output_layer=1))
        elif self.model_type == 'gcn' or self.model_type == 'acmgcn':
            self.gcns.append(GraphConvolution_acm(
                nfeat, nhid,  model_type=model_type, variant=variant))
            self.gcns.append(GraphConvolution_acm(
                nhid, nclass,  model_type=model_type, output_layer=1, variant=variant))
        elif self.model_type == 'sgc' or self.model_type == 'acmsgc':
            self.gcns.append(GraphConvolution_acm(
                nfeat, nclass, model_type=model_type))
        elif self.model_type == 'acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolution_acm(
                    k * nhid + nfeat, nhid, model_type=model_type, variant=variant))
            self.gcns.append(GraphConvolution_acm(
                nlayers * nhid + nfeat, nclass, model_type=model_type, variant=variant))
        self.dropout = dropout

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, x, adj_low, adj_high):
        if self.model_type == 'acmgcn' or self.model_type == 'acmsgc' or self.model_type == 'acmsnowball':
            x = F.dropout(x, self.dropout, training=self.training)

        if self.model_type == 'acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(
                        F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat(
                        [x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))
            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea = (self.gcns[0](x, adj_low, adj_high))

        if self.model_type == 'gcn' or self.model_type == 'mlp' or self.model_type == 'acmgcn':
            fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
            fea = self.gcns[-1](fea, adj_low, adj_high)
        # return fea
        return F.log_softmax(fea, dim=1), F.softmax(fea, dim=1)
    
class GloGNN(nn.Module):
    def __init__(self, args):
    # def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta,
                 # norm_func_id, norm_layers, orders, orders_func_id, device):
        super(GloGNN, self).__init__()
        
        nnodes = args.num_nodes
        nfeat = args.num_features
        nhid = args.hidden
        nclass = args.C
        dropout = args.dropout
        alpha = args.glognn_alpha
        beta1 = args.glognn_beta1
        beta2 = args.glognn_beta2
        gamma = args.glognn_gamma
        norm_func_id = args.norm_func_id
        norm_layers = args.norm_layers
        orders = args.orders
        orders_func_id = args.orders_func_id
        device = "cuda:"+str(args.cuda)
        
        self.fc1 = nn.Linear(nfeat, nhid).to(torch.float64)
        self.fc2 = nn.Linear(nhid, nclass).to(torch.float64)
        self.fc3 = nn.Linear(nhid, nhid).to(torch.float64)
        self.fc4 = nn.Linear(nnodes, nhid).to(torch.float64)
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha)
        self.beta1 = torch.tensor(beta1)
        self.beta2 = torch.tensor(beta2)
        self.gamma = torch.tensor(gamma)
        # self.alpha = torch.tensor(alpha).to(device)
        # self.beta = torch.tensor(beta).to(device)
        # self.gamma = torch.tensor(gamma).to(device)
        # self.delta = torch.tensor(delta).to(device)
        self.norm_layers = norm_layers
        self.orders = orders
        # self.device = device
        # self.class_eye = torch.eye(self.nclass)
        self.class_eye = torch.eye(self.nclass).to(device)
        self.orders_weight = Parameter(
            (torch.ones(orders, 1) / orders), requires_grad=True
            # (torch.ones(orders, 1) / orders).to(device), requires_grad=True
        ).to(device).to(torch.float64)
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(nclass, orders), requires_grad=True
            # torch.DoubleTensor(nclass, orders).to(device), requires_grad=True
        ).to(device)
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(orders, orders), requires_grad=True
            # torch.DoubleTensor(orders, orders).to(device), requires_grad=True
        ).to(device)
        self.diag_weight = Parameter(
            (torch.ones(nclass, 1) / nclass), requires_grad=True
            # (torch.ones(nclass, 1) / nclass).to(device), requires_grad=True
        ).to(device).to(torch.float64)
        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.orders_weight = Parameter(
            (torch.ones(self.orders, 1) / self.orders), requires_grad=True
            # (torch.ones(self.orders, 1) / self.orders).to(self.device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.diag_weight = Parameter(
            (torch.ones(self.nclass, 1) / self.nclass), requires_grad=True
            # (torch.ones(self.nclass, 1) / self.nclass).to(self.device), requires_grad=True
        )

    def forward(self, x, adj):
        xX = self.fc1(x)
        xA = self.fc4(adj)
        x = F.relu(self.alpha * xX + (1-self.alpha) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        # return x
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)

    def norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.beta1 + self.beta2)
        coe1 = 1.0 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta2 * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.beta1 + self.beta2)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta2 * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        orders_para = torch.transpose(orders_para, 0, 1)
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders
    
    
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

# the same as GCN but remove trainable weights
class DGCNConv(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, 
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(DGCNConv, self).__init__(**kwargs)

        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class DGCN(nn.Module):
    # def __init__(self, input_dim, out_dim, filter_num, dropout = False, layer = 2):
    def __init__(self, args):
        super(DGCN, self).__init__()
        input_dim = args.num_features
        out_dim = args.C
        filter_num = args.hidden
        layer = 2
        
        self.dropout = args.dropout
        self.gconv = DGCNConv()
        self.Conv = nn.Conv1d(filter_num*3, out_dim, kernel_size=1)

        self.lin1 = torch.nn.Linear(input_dim,    filter_num,   bias=False)
        self.lin2 = torch.nn.Linear(filter_num*3, filter_num, bias=False)

        self.bias1 = nn.Parameter(torch.Tensor(1, filter_num))
        self.bias2 = nn.Parameter(torch.Tensor(1, filter_num))

        self.layer = layer
        if layer == 3:
            self.lin3 = torch.nn.Linear(filter_num*3, filter_num, bias=False)
            self.bias3 = nn.Parameter(torch.Tensor(1, filter_num))
            nn.init.zeros_(self.bias3)

        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)

    def forward(self, x, edge_index, edge_in, in_w, edge_out, out_w):
        x = self.lin1(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)
        
        x1 += self.bias1
        x2 += self.bias1
        x3 += self.bias1

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        x = self.lin2(x)
        x1 = self.gconv(x, edge_index)
        x2 = self.gconv(x, edge_in, in_w)
        x3 = self.gconv(x, edge_out, out_w)

        x1 += self.bias2
        x2 += self.bias2
        x3 += self.bias2

        x = torch.cat((x1, x2, x3), axis = -1)
        x = F.relu(x)

        if self.layer == 3:
            x = self.lin3(x)
            x1 = self.gconv(x, edge_index)
            x2 = self.gconv(x, edge_in, in_w)
            x3 = self.gconv(x, edge_out, out_w)

            x1 += self.bias3
            x2 += self.bias3
            x3 += self.bias3

            x = torch.cat((x1, x2, x3), axis = -1)
            x = F.relu(x)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()

        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)
    
if __name__ == '__main__':
    pass