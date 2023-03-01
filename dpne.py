import math
import torch
import numpy as np
import scipy.sparse as sp
from torch.autograd import Variable

from torch_geometric import datasets
from torch_geometric.utils import to_dense_adj


def get_fact_mat(adj):
    print("compute the factorization matrix M.")
    X = np.copy(adj)
    d_rt = np.sum(X, axis=1, dtype=np.float32)
    D_rt_inv = sp.diags(d_rt ** -1)
    D_inv = D_rt_inv.toarray()
    vol_G = d_rt.sum()

    X = np.matmul(D_inv, X)
    X_power = np.eye(X.shape[0], dtype=np.float32)
    S = np.zeros_like(X)
    for i in range(10):
        X_power = np.matmul(X, X_power)
        S += X_power
    output = S * (vol_G / 10 / 5)
    output = np.matmul(output, D_inv)
    output = np.clip(output, a_min=1, a_max=np.inf)
    output = np.log(output)

    return output


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = datasets.Planetoid('datasets', 'Cora')[0]
    adj = to_dense_adj(dataset.edge_index).numpy()[0]
    n_samples = adj.shape[0]
    dim = 256
    M = get_fact_mat(adj)
    learning_rate = 1e-4
    beta = 0.001
    epsilon = 0
    if epsilon == 0:
        M_torch = torch.FloatTensor(M).to(device)
        H = torch.randn([dim, n_samples], device=device)
        H = H / H.max()
        H = Variable(H, requires_grad=True)

        W = torch.randn([n_samples, dim], device=device)
        W = W / W.max()
        W = Variable(W, requires_grad=True)
        WH = torch.mm(W, H)
        optimizer = torch.optim.Adam([W, H], lr=learning_rate)

        for i in range(2000):
            loss = (M_torch - WH).pow(2).sum() + beta * (H.pow(2).sum() + W.pow(2).sum())
            loss.backward()
            optimizer.step()
            WH = torch.mm(W, H)
            if (i + 1) % 100 == 0:
                print(i+1, loss.cpu().data.numpy())
    else:
        mask = np.zeros((n_samples, dim)).astype(np.float32)
        for i in range(n_samples):
            norm = 1 / (0 + 1) * np.random.exponential(1.0 / epsilon, dim)
            b = (1 / math.sqrt(dim)) * np.random.normal(0, 1, dim)
            noise = (sum(norm) / math.sqrt(sum(pow(b, 2)))) * b
            mask[i] = noise

        M_torch = Variable(torch.FloatTensor(M))
        mask_torch = Variable(torch.FloatTensor(mask))
        H = torch.randn([dim, n_samples])
        H = H / H.max()
        H = Variable(H, requires_grad=True)

        W = torch.randn([n_samples, dim])
        W = W / W.max()
        W = Variable(W, requires_grad=True)
        WH = torch.mm(W, H)
        optimizer = torch.optim.Adam([W, H], lr=learning_rate)

        for i in range(1000):
            loss = (M_torch - WH).pow(2).sum() + beta * (H.pow(2).sum() + W.pow(2).sum()) + abs((mask_torch * W).sum())
            loss.backward()
            optimizer.step()
            WH = torch.mm(W, H)
            if (i + 1) % 100 == 0:
                print(i+1, loss.data.numpy())
