
# from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from typing import Any, Callable, Dict, List, Optional, Union
import warnings
import inspect
import math
import torch
from torch import Tensor
from torch.nn import Parameter, Identity
import torch.nn.functional as F
from torch_sparse import spmm, SparseTensor, matmul
from torch_geometric.nn import inits
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, NoneType

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


class SparseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """

    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(
            self.in_channels, self.out_channels)).to(DEVICE)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(DEVICE)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        # feature_count, _ = torch.max(features["indices"], dim=1)
        # feature_count = feature_count + 1

        # base_features = spmm(features["indices"], features["values"], feature_count[0],
        #                      feature_count[1], self.weight_matrix)
        base_features = spmm(features["indices"].to(DEVICE), features["values"].to(DEVICE), features["dimensions"][0],
                             features["dimensions"][1], self.weight_matrix)

        base_features = base_features + self.bias

        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)

        base_features = torch.nn.functional.relu(base_features)
        for _ in range(self.iterations - 1):
            base_features = spmm(normalized_adjacency_matrix["indices"].to(DEVICE),
                                 normalized_adjacency_matrix["values"].to(DEVICE),
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        return base_features


class DenseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Dense Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """

    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(
            self.in_channels, self.out_channels)).to(DEVICE)
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels)).to(DEVICE)

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.mm(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features,
                                                    p=self.dropout_rate,
                                                    training=self.training)
        for _ in range(self.iterations - 1):
            base_features = spmm(normalized_adjacency_matrix["indices"].to(DEVICE),
                                 normalized_adjacency_matrix["values"].to(DEVICE),
                                 base_features.shape[0],
                                 base_features.shape[0],
                                 base_features)
        base_features = base_features + self.bias
        return base_features


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """

    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class MLP(torch.nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class SparseLinear(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.kaiming_uniform(self.weight, fan=self.in_channels,
                              a=math.sqrt(5))
        inits.uniform(self.in_channels, self.bias)

    def forward(self, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # propagate_type: (weight: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, weight=self.weight,
                             edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, weight_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return weight_j
        else:
            return edge_weight.view(-1, 1) * weight_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              weight: Tensor) -> Tensor:
        return matmul(adj_t, weight, reduce=self.aggr)


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: List[Any], class_dict: Dict[str, Any],
             query: Union[Any, str], base_cls: Optional[Any], *args, **kwargs):

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    base_cls_repr = normalize_string(base_cls.__name__) if base_cls else ''

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


# Activation Resolver #########################################################


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs):
    import torch
    base_cls = torch.nn.Module
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, *args, **kwargs)


# Normalization Resolver ######################################################


def normalization_resolver(query: Union[Any, str], *args, **kwargs):
    import torch

    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, *args, **kwargs)


## from https://github.com/RecklessRonan/GloGNN

class GraphConvolution_acm(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, model_type, output_layer=0, variant=False):
        super(GraphConvolution_acm, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type, self.variant = in_features, out_features, output_layer, model_type, variant
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        if torch.cuda.is_available():
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features).cuda()), Parameter(
                torch.FloatTensor(in_features, out_features).cuda()), Parameter(torch.FloatTensor(in_features, out_features).cuda())
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1).cuda(
            )), Parameter(torch.FloatTensor(out_features, 1).cuda()), Parameter(torch.FloatTensor(out_features, 1).cuda())
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(1, 1).cuda(
            )), Parameter(torch.FloatTensor(1, 1).cuda()), Parameter(torch.FloatTensor(1, 1).cuda())

            self.att_vec = Parameter(torch.FloatTensor(3, 3).cuda())

        else:
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features)), Parameter(
                torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1)), Parameter(
                torch.FloatTensor(out_features, 1)), Parameter(torch.FloatTensor(out_features, 1))
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(
                1, 1)), Parameter(torch.FloatTensor(1, 1)), Parameter(torch.FloatTensor(1, 1))

            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))

        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm(
            (output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)], 1)), self.att_vec)/T, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high):
        output = 0
        if self.model_type == 'mlp':
            output_mlp = (torch.mm(input, self.weight_mlp))
            return output_mlp
        elif self.model_type == 'sgc' or self.model_type == 'gcn':
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == 'acmgcn' or self.model_type == 'acmsnowball':
            if self.variant:
                output_low = (torch.spmm(adj_low, F.relu(
                    torch.mm(input, self.weight_low))))
                output_high = (torch.spmm(adj_high, F.relu(
                    torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))
            else:
                output_low = F.relu(torch.spmm(
                    adj_low, (torch.mm(input, self.weight_low))))
                output_high = F.relu(torch.spmm(
                    adj_high, (torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 3*(output_low + output_high + output_mlp) #
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
        elif self.model_type == 'acmsgc':
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            # torch.mm(input, self.weight_high) - torch.spmm(self.A_EXP,  torch.mm(input, self.weight_high))
            output_high = torch.spmm(
                adj_high,  torch.mm(input, self.weight_high))
            output_mlp = torch.mm(input, self.weight_mlp)

            # self.attention(F.relu(output_low), F.relu(output_high), F.relu(output_mlp))
            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 3*(output_low + output_high + output_mlp) #
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
