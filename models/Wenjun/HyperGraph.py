from pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time
import math
import numpy as np
import os
# from .torch_nn_hypergraph import norm_layer, act_layer
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


def pairwise_euclidean_distance(x: torch.Tensor):
    """
    generate N x N node distance matrix
    :param x: a tensor of size N x C (N nodes with C feature dimension)
    :return: a tensor of N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 2
    x = x.float()

    x_transpose = torch.transpose(x, dim0=0, dim1=1)
    x_inner = torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    x_square_transpose = torch.transpose(x_square, dim0=0, dim1=1)
    dis = x_square + x_inner + x_square_transpose
    return dis

def pairwise_euclidean_distance_dis(x: torch.Tensor):
    """
    generate B x N x N node distance matrix
    :param x: a tensor of size B x N x C (N nodes with C feature dimension)
    :return: a tensor of B x N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 3
    x = x.float()

    x_transpose = torch.transpose(x, dim0=1, dim1=2)
    x_inner = torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=2, keepdim=True)
    x_square_transpose = torch.transpose(x_square, dim0=1, dim1=2)
    dis = x_square + x_inner + x_square_transpose
    return dis    

def neighbor_distance(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: N x C matrix. N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """

    assert len(x.shape) == 2, 'should be a tensor with dimension (N x C)'
    #print("-" * 20)
    #print("x", x.device)
    # N x C
    node_num = x.size(0)
    dis_matrix = dis_metric(x)
    #print("dis_matrix:", dis_matrix.shape)
    # print("-" * 20)
    # print("dis_matrix", dis_matrix.device)
    _, nn_idx = torch.topk(dis_matrix, k_nearest-1, dim=1, largest=False)
    # print("-" * 20)
    # print("nn_idx", nn_idx.device)
    self_node = torch.arange(node_num).unsqueeze(dim=1).to(x.device)
    # print("-" * 20)
    # print("self_node", self_node.device)
    nn_idx = torch.cat((nn_idx, self_node), 1)
    # print("-" * 20)
    # print("nn_idx", type(nn_idx))
    nn_idx = nn_idx.reshape(-1)
    # print("-" * 20)
    # print("nn_idx", nn_idx.shape)
    hyedge_idx = torch.arange(node_num).to(x.device).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1)
    # print("hyedge_idx", hyedge_idx.shape)
    h = torch.zeros(node_num, node_num)
    index = (nn_idx,hyedge_idx)#生成索引
    value = torch.Tensor([1]) #生成要填充的值
    h.index_put_(index, value)
    return h.to(x.device)

def neighbor_distance_dis(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance_dis):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: B x N x C matrix. B denotes batch size, N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """

    assert len(x.shape) == 3, 'should be a tensor with dimension (B x N x C)'
    B, N, C = x.shape
    node_num = N
    dis_matrix = dis_metric(x) # (B x N x N)
    # print("dis_matrix:", dis_matrix.shape)
    # print("-" * 20)
    # print("dis_matrix", dis_matrix.device)
    _, nn_idx = torch.topk(dis_matrix, k_nearest-1, dim=2, largest=False)
    # print("-" * 20)
    # print("nn_idx", nn_idx.shape)
    self_node = torch.arange(node_num).unsqueeze(dim=1).repeat(B,1,1).to(x.device)
    # print("-" * 20)
    # print("self_node", self_node.device)
    nn_idx = torch.cat((nn_idx, self_node), 2)
    # print("-" * 20)
    # print("nn_idx", type(nn_idx))
    nn_idx = nn_idx.reshape(-1)
    # print("-" * 20)
    # print("nn_idx", type(nn_idx))
    hyedge_idx = torch.arange(node_num).to(x.device).unsqueeze(0).repeat(B, k_nearest, 1).transpose(2, 1).reshape(-1)
    # H = torch.stack([nn_idx.reshape(-1), hyedge_idx])
    h = torch.zeros(B, node_num, node_num)
    batch_index = torch.arange(B).unsqueeze(1).repeat(1,N*k_nearest).reshape(-1)
    index = (batch_index, nn_idx, hyedge_idx)#生成索引

    # print("nn_idx", nn_idx.shape)   
    # print("hyedge_idx", hyedge_idx.shape)
    # print(torch.stack([nn_idx,hyedge_idx],dim=-1).shape)
    value = torch.Tensor([1]) #生成要填充的值
    # print(index)
    # h[[[0,0]]] = value
    h.index_put_(index, value)
    return h.to(x.device)

def degree_node(H, node_num=None):
    tmp = torch.sum(H, dim=1)
    degree_node_matrix = torch.diag(tmp)
    return degree_node_matrix

def degree_node_dis(H, node_num=None):
    tmp = torch.sum(H, dim=2)
    degree_node_matrix = torch.diag_embed(tmp)
    return degree_node_matrix

def degree_hyedge(H: torch.Tensor, hyedge_num=None):
    tmp = torch.sum(H, dim=0)
    degree_hyedge_matrix = torch.diag(tmp)
    return degree_hyedge_matrix

def degree_hyedge_dis(H: torch.Tensor, hyedge_num=None):
    tmp = torch.sum(H, dim=1)
    degree_hyedge_matrix = torch.diag_embed(tmp)
    return degree_hyedge_matrix

class DAHHConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias=None) -> None:
        super().__init__()
        # self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        self.theta = theta
        self.bias = bias
        # if bias:
        #     self.bias = Parameter(torch.Tensor(out_ch))
        # else:
        #     self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def gen_hyedge_ft(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        hyedge_num = count_hyedge(H)

        # a vector to normalize hyperedge feature
        hyedge_norm = 1.0 / degree_hyedge(H).float()
        if hyedge_weight is not None:
            hyedge_norm *= hyedge_weight
        hyedge_norm = hyedge_norm[hyedge_idx.long()]

        x = x[node_idx.long()] * hyedge_norm.unsqueeze(1)
        x = torch.zeros(hyedge_num, ft_dim).to(x.device).scatter_add(0, hyedge_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_hyedge_matrix = degree_hyedge(H)
        hyedge_norm = degree_hyedge_matrix.inverse()
        # hyedge_norm = degree_hyedge_matrix
        # for i in range(degree_hyedge_matrix.shape[0]):
            # hyedge_norm[i] = degree_hyedge_matrix[i].inverse
        tmp = torch.matmul(H, hyedge_norm)
        return torch.matmul(tmp.T, x)

    def gen_node_ft(self, x: torch.Tensor, H: torch.Tensor):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        node_num = count_node(H)

        # a vector to normalize node feature
        node_norm = 1.0 / degree_node(H).float()
        node_norm = node_norm[node_idx]

        x = x[hyedge_idx] * node_norm.unsqueeze(1)
        x = torch.zeros(node_num, ft_dim).to(x.device).scatter_add(0, node_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_node_matrix = degree_node(H).to(x.device)
        # node_norm = degree_node_matrix
        # for i in range(degree_node_matrix.shape[0]):
        #     node_norm[i] = degree_node_matrix[i].inverse
        node_norm = degree_node_matrix.inverse()
        
        
        # print("node_norm2", node_norm2)
        tmp = torch.matmul(node_norm, H)
        return torch.matmul(tmp, x)

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        assert len(x.shape) == 2, 'the input of HyperConv should be N x C'
        # feature transform
        
        x = x.matmul(self.theta)

        # generate hyperedge feature from node feature
        x = self.gen_hyedge_ft(x, H, hyedge_weight)
        
        # generate node feature from hyperedge feature
        x = self.gen_node_ft(x, H)
        
        if self.bias is not None:
            return x + self.bias
        else:
            return x

class DAHHConv_dis(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        # self.theta = theta
        # self.bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def gen_hyedge_ft(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        hyedge_num = count_hyedge(H)

        # a vector to normalize hyperedge feature
        hyedge_norm = 1.0 / degree_hyedge(H).float()
        if hyedge_weight is not None:
            hyedge_norm *= hyedge_weight
        hyedge_norm = hyedge_norm[hyedge_idx.long()]

        x = x[node_idx.long()] * hyedge_norm.unsqueeze(1)
        x = torch.zeros(hyedge_num, ft_dim).to(x.device).scatter_add(0, hyedge_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_hyedge_matrix = degree_hyedge_dis(H)
        # hyedge_norm = degree_hyedge_matrix
        # for i in range(degree_hyedge_matrix.shape[0]):
            # hyedge_norm[i] = degree_hyedge_matrix[i].inverse()
        
        hyedge_norm = degree_hyedge_matrix.inverse()
        tmp = torch.matmul(H, hyedge_norm)
        tmp = torch.transpose(tmp, dim0=1, dim1=2)
        return torch.matmul(tmp, x)

    def gen_node_ft(self, x: torch.Tensor, H: torch.Tensor):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        node_num = count_node(H)

        # a vector to normalize node feature
        node_norm = 1.0 / degree_node(H).float()
        node_norm = node_norm[node_idx]

        x = x[hyedge_idx] * node_norm.unsqueeze(1)
        x = torch.zeros(node_num, ft_dim).to(x.device).scatter_add(0, node_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_node_matrix = degree_node_dis(H).to(x.device)
        # node_norm = degree_node_matrix
        # for i in range(degree_node_matrix.shape[0]):
            # node_norm[i] = degree_node_matrix[i].inverse()
        node_norm = degree_node_matrix.inverse()
        
        
        # print("node_norm2", node_norm2)
        tmp = torch.matmul(node_norm, H)
        return torch.matmul(tmp, x)

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        assert len(x.shape) == 3, 'the input of HyperConv should be B x N x C'
        # feature transform
        # B, N, C = x.shape
        # theta = Parameter(torch.Tensor(B, in_ch, out_ch))
       
        x = x.matmul(self.theta)
        
        # generate hyperedge feature from node feature
        x = self.gen_hyedge_ft(x, H, hyedge_weight)
        
        # generate node feature from hyperedge feature
        x = self.gen_node_ft(x, H)

        if self.bias is not None:
            return x + self.bias
        else:
            return x

class DAHHConv_flatten(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True) -> None:
        super().__init__()
        # self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        self.theta = theta
        self.bias = bias
        # if bias:
            # self.bias = Parameter(torch.Tensor(out_ch))
        # else:
            # self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def gen_hyedge_ft(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        hyedge_num = count_hyedge(H)

        # a vector to normalize hyperedge feature
        hyedge_norm = 1.0 / degree_hyedge(H).float()
        if hyedge_weight is not None:
            hyedge_norm *= hyedge_weight
        hyedge_norm = hyedge_norm[hyedge_idx.long()]

        x = x[node_idx.long()] * hyedge_norm.unsqueeze(1)
        x = torch.zeros(hyedge_num, ft_dim).to(x.device).scatter_add(0, hyedge_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_hyedge_matrix = degree_hyedge_dis(H)#.reshape(4, 1, 3000, 3000).contiguous()
        hyedge_norm = degree_hyedge_matrix
        for i in range(degree_hyedge_matrix.shape[0]):
            hyedge_norm[i] = degree_hyedge_matrix[i].inverse()
        # print(degree_hyedge_matrix.shape)
        # hyedge_norm = degree_hyedge_matrix.inverse()#.reshape(4, 3000, 3000).contiguous()
        tmp = torch.matmul(H, hyedge_norm)
        tmp = torch.transpose(tmp, dim0=1, dim1=2)
        return torch.matmul(tmp, x)

    def gen_node_ft(self, x: torch.Tensor, H: torch.Tensor):
        '''
        ft_dim = x.size(1)
        node_idx, hyedge_idx = H
        node_num = count_node(H)

        # a vector to normalize node feature
        node_norm = 1.0 / degree_node(H).float()
        node_norm = node_norm[node_idx]

        x = x[hyedge_idx] * node_norm.unsqueeze(1)
        x = torch.zeros(node_num, ft_dim).to(x.device).scatter_add(0, node_idx.unsqueeze(1).repeat(1, ft_dim), x)
        return x
        '''
        degree_node_matrix = degree_node_dis(H)#.reshape(4, 1, 3000, 3000).contiguous()
        node_norm = degree_node_matrix
        for i in range(degree_node_matrix.shape[0]):
            node_norm[i] = degree_node_matrix[i].inverse()
        # print("node:",degree_node_matrix.shape)
        # node_norm = degree_node_matrix.inverse().reshape(4, 3000, 3000).contiguous()
        
        
        # print("node_norm2", node_norm2)
        tmp = torch.matmul(node_norm, H)
        return torch.matmul(tmp, x)

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyedge_weight=None):
        assert len(x.shape) == 3, 'the input of HyperConv should be B x N x C'
        # feature transform
        # B, N, C = x.shape
        # theta = Parameter(torch.Tensor(B, in_ch, out_ch))
       
        x = x.matmul(self.theta)
        
        # generate hyperedge feature from node feature
        x = self.gen_hyedge_ft(x, H, hyedge_weight)
        
        # generate node feature from hyperedge feature
        x = self.gen_node_ft(x, H)

        if self.bias is not None:
            return x + self.bias
        else:
            return x

class DAHH(nn.Module):
    def __init__(self, in_channels=768, out_channels=159, act='relu', norm=None, bias=True, drop=0.):
        super(DAHH, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.norm = norm
        self.bias = bias
        self.drop = drop
        self.hconv = DAHHConv(self.in_channels, self.out_channels) # c-> 2c
        # if self.norm is not None and self.norm.lower() != 'none':
        #     self.norm = norm_layer(self.norm, self.out_channels)
        self.norm = nn.BatchNorm2d(self.out_channels)
        if self.act is not None and self.act.lower() != 'none':
            self.act = act_layer(self.act)
        if self.drop > 0:
            self.drop = nn.Dropout2d(self.drop)


    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B, L, -1).contiguous()
        for i in range(x.shape[0]):
            H = neighbor_distance(x[i], 3)    # H : 196 * 196
            res = self.hconv(x[i], H)
            if i==0:
                out = res
            else:
                out = torch.cat((out,res), dim=0)
        L = out.shape[0] // x.shape[0]
        x = out.view(B, -1, L, 1)
        # x = out.view(B,L,-1)
        # print(x.shape)
        x = self.norm(x)
        x = self.act(x)
        if self.drop != 0.:
            x = self.drop(x)
        return x

class HyperGraphBlock(nn.Module):
    def __init__(self, in_channels=768, out_channels=159, act='relu', norm=None, bias=True, drop=0.):
        super(HyperGraphBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.norm = norm
        self.bias = bias
        self.drop = drop
        self.hconv = DAHHConv_dis(self.in_channels, self.out_channels) # c-> 2c
        self.norm = nn.BatchNorm2d(self.out_channels)
        if self.act is not None and self.act.lower() != 'none':
            self.act = act_layer(self.act)
        if self.drop > 0:
            self.drop = nn.Dropout2d(self.drop)


    def forward(self, x):
        B, L, C = x.shape
        # x = x.reshape(B, L, -1).contiguous()
        H = neighbor_distance_dis(x, 3)
        x = self.hconv(x,H)
        x = x.view(B, -1, L, 1)
        x = self.norm(x)
        x = self.act(x)
        if self.drop != 0.:
            x = self.drop(x)
        x = x.reshape(B, L, -1)
        return x
class DAHH_flatten(nn.Module):
    def __init__(self, in_channels=768, out_channels=159, act='relu', norm=None, bias=True, drop=0.):
        super(DAHH_flatten, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.norm = norm
        self.bias = bias
        self.drop = drop
        self.hconv = DAHHConv_flatten(self.in_channels, self.out_channels) # c-> 2c
        self.norm = nn.BatchNorm2d(self.out_channels)
        if self.act is not None and self.act.lower() != 'none':
            self.act = act_layer(self.act)
        if self.drop > 0:
            self.drop = nn.Dropout2d(self.drop)


    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B, L, -1).contiguous()
        H = neighbor_distance_dis(x, 3)
        x = self.hconv(x,H)
        x = x.view(B, -1, L, 1)
        x = self.norm(x)
        x = self.act(x)
        if self.drop != 0.:
            x = self.drop(x)
        return x


if __name__ == "__main__":
    x = torch.randn(4, 1024, 3)
    ft = torch.rand(4, 1024, 64)
    model = HyperGraphBlock(3,6)
    y = model(x)
    print(y.shape)