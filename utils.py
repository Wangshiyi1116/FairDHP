from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter
import random
import torch
import os
from sklearn.cluster  import KMeans
from sklearn.preprocessing  import Normalizer
import time
import numpy as np
import scipy.sparse as sp
from vector_quantize_pytorch import VectorQuantize
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
import pandas as pd


def kmeans_projection(features, n_clusters=100, metric='euclidean'):
    """执行K-means并生成投影矩阵"""
    # 预处理：余弦距离需L2归一化
    if metric == 'cosine':
        normalizer = Normalizer(norm='l2')
        features = normalizer.transform(features)

        # 执行K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(features)

    # 生成one-hot投影矩阵
    projection = np.eye(n_clusters)[labels]
    return projection

def propagate(x, edge_index, edge_weight=None):
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    if(edge_weight == None):
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')

def propagate2(x, edge_index):
    edge_index, _ = add_remaining_self_loops(
        edge_index, num_nodes=x.size(0))

    # calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # normalize the features on the starting point of the edge
    out = edge_weight.view(-1, 1) * x[row]

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.allow_tf32 = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    # torch.use_deterministic_algorithms(True)


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) -
                 sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) -
                   sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()

def random_drop_edges(adj, drop_prob):
    mask = torch.rand(adj.size()) > drop_prob
    adj = adj * mask
    adj = adj + adj.t() - adj * adj.t()
    return adj

def log_diff(f1, f2, isdiag = True):
    abs_diff = torch.abs(f1 - f2)
    diff = torch.log(abs_diff.sum(dim=1, keepdim=True) + 1e-6).squeeze()
    diff[diff<0] = 0.
    max = diff.max()
    diff = diff / max
    if isdiag == True:
        size = diff.shape[0]
        rows = torch.arange(size)
        cols = torch.arange(size)
        values = diff.cpu()
        indices = torch.stack([rows, cols], dim=0)
        # 创建稀疏的对角矩阵 (COO 格式)
        sparse_diag = torch.sparse.FloatTensor(indices, values, torch.Size([size, size]))
        return sparse_diag
        # return torch.diag(diff)
    else:
        return diff

def VQ_adj(features, dim, codebook_size, decay=0.8, commitment_weight=1):
    # start = time.perf_counter()
    # proj_euclidean = kmeans_projection(features.cpu(), metric='euclidean')
    # time_euclidean = time.perf_counter() - start
    # start = time.perf_counter()
    # proj_cosine = kmeans_projection(features.cpu(), metric='cosine')
    # time_cosine = time.perf_counter() - start
    # print(f"欧式投影矩阵| 耗时: {time_euclidean:.4f}s")
    # print(f"余弦投影矩阵| 耗时: {time_cosine:.4f}s")
    #
    # start = time.perf_counter()
    # proj_euclidean = VQ_compond(features, dim, codebook_size, False, decay, commitment_weight)
    # time_r1 = time.perf_counter() - start
    # start = time.perf_counter()
    # proj_cosine = VQ_compond(features, dim, codebook_size, True, decay, commitment_weight)
    # time_r2 = time.perf_counter() - start
    # print(f"r1| 耗时: {time_r1:.4f}s")
    # print(f"r2| 耗时: {time_r2:.4f}s")

    R1, _ = VQ_compond(features, dim, codebook_size, False, decay, commitment_weight)
    R2, _ = VQ_compond(features, dim, codebook_size, True, decay, commitment_weight)
    R = torch.concat((R1,R2),dim=1)
    try:
        adj = R@R.T
    except:
        R = R.coalesce()
        indices = R.indices()
        values = R.values()
        n, m = R.size()
        adj_indices = []
        adj_values = []
        for k in range(values.size(0)):
            i = indices[0, k].item()
            j = indices[1, k].item()
            value = values[k].item()
            adj_indices.append([i, j])
            adj_values.append(value)
            if i != j:
                adj_indices.append([j, i])
                adj_values.append(value)
        adj_indices = torch.tensor(adj_indices, device='cuda:0').t()
        adj_values = torch.tensor(adj_values, device='cuda:0')
        adj = torch.sparse_coo_tensor(adj_indices, adj_values, (n, n), device='cuda:0')
    return adj

def VQ_compond(features, dim, codebook_size, use_cosine_sim,decay=0.8,commitment_weight=1):
    vq = VectorQuantize(dim=dim,
                        codebook_size=codebook_size,
                        decay=decay,
                        commitment_weight=commitment_weight,
                        use_cosine_sim=use_cosine_sim).cuda()
    quantized, indices, _ = vq(features)
    quantized = quantized.cuda()
    indices = indices.cuda()
    rows = torch.arange(indices.shape[0]).cuda()
    cols = indices.squeeze().cuda()
    values = torch.ones(indices.shape[0]).cuda()
    R = torch.sparse_coo_tensor(torch.vstack([rows, cols]), values, (indices.shape[0], indices.max() + 1)).cuda()
    return R, quantized



def to_sparse_tensor(matrix):
    """
    Convert a dense PyTorch tensor, a NumPy array, or a SciPy sparse matrix
    to a PyTorch sparse tensor.

    :param matrix: (Tensor, ndarray, or scipy.sparse matrix) Input matrix.
    :return: (torch.sparse_coo_tensor) PyTorch sparse tensor.
    """
    if isinstance(matrix, torch.Tensor):
        # If already a dense tensor, convert to sparse
        if matrix.is_sparse:
            return matrix
        else:
            indices = torch.nonzero(matrix).t()
            values = matrix[indices[0], indices[1]]
            return torch.sparse_coo_tensor(indices, values, matrix.size())

    elif isinstance(matrix, np.ndarray):
        # Convert NumPy array to dense tensor first
        matrix_tensor = torch.tensor(matrix)
        indices = torch.nonzero(matrix_tensor).t()
        values = matrix_tensor[indices[0], indices[1]]
        return torch.sparse_coo_tensor(indices, values, matrix_tensor.size())

    elif sp.issparse(matrix):
        # For SciPy sparse matrices
        matrix = matrix.tocoo()  # Ensure COO format
        indices = torch.tensor([matrix.row, matrix.col], dtype=torch.long)
        values = torch.tensor(matrix.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, matrix.shape)

    else:
        raise TypeError("Input must be a PyTorch Tensor, NumPy array, or SciPy sparse matrix.")
