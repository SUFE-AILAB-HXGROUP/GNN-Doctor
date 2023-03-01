import numpy as np
import scipy.sparse as sp


def lap_graph(adj, epsilon=10, noise_type='laplace', noise_seed=42, delta=1e-5):
    """
    :param adj: sparse adjacency matrix
    :param epsilon: param for differential privacy
    :param noise_type: laplace noise
    :param noise_seed: noise seed
    :param delta:
    :return:
    """
    n_nodes = adj.shape[0]
    n_edges = adj.sum() / 2

    N = n_nodes

    A = sp.tril(adj, k=-1)

    eps_1 = epsilon * 0.01
    eps_2 = epsilon - eps_1
    noise = get_noise(noise_type=noise_type, size=(N, N), seed=noise_seed,
                      eps=eps_2, delta=delta, sensitivity=1)
    noise *= np.tri(*noise.shape, k=-1, dtype=bool)

    A += noise

    while True:
        n_edges_keep = int(n_edges + int(
            get_noise(noise_type=noise_type, size=1, seed=noise_seed,
                      eps=eps_1, delta=delta, sensitivity=1)[0]))
        if n_edges_keep > 0:
            break
        else:
            continue

    a_r = A.A.ravel()

    n_splits = 50
    len_h = len(a_r) // n_splits
    ind_list = []
    for i in range(n_splits - 1):
        ind = np.argpartition(a_r[len_h * i:len_h * (i + 1)], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * i)

    ind = np.argpartition(a_r[len_h * (n_splits - 1):], -n_edges_keep)[-n_edges_keep:]
    ind_list.append(ind + len_h * (n_splits - 1))

    ind_subset = np.hstack(ind_list)
    a_subset = a_r[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

    row_idx = []
    col_idx = []
    for idx in ind:
        idx = ind_subset[idx]
        row_idx.append(idx // N)
        col_idx.append(idx % N)
        assert (col_idx < row_idx)
    data_idx = np.ones(n_edges_keep, dtype=np.int32)

    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
    return mat + mat.T


def edge_rand(adj, epsilon=10, noise_seed=42):
    """
    :param adj: sparse adjacency matrix
    :param epsilon: param for differential privacy
    :param noise_seed: noise seed
    :return:
    """
    s = 2 / (np.exp(epsilon) + 1)

    N = adj.shape[0]

    # np.random.seed(noise_seed)
    bernoulli = np.random.binomial(1, s, (N, N))

    # find the randomization entries
    entry = np.asarray(list(zip(*np.where(bernoulli))))

    dig_1 = np.random.binomial(1, 1 / 2, len(entry))
    indice_1 = entry[np.where(dig_1 == 1)[0]]
    indice_0 = entry[np.where(dig_1 == 0)[0]]

    add_mat = construct_sparse_mat(indice_1, N)
    minus_mat = construct_sparse_mat(indice_0, N)

    adj_noisy = adj + add_mat - minus_mat

    adj_noisy.data[np.where(adj_noisy.data == -1)] = 0
    adj_noisy.data[np.where(adj_noisy.data == 2)] = 1

    return adj_noisy


def get_noise(noise_type, size, seed, eps=10.0, delta=1e-5, sensitivity=2):
    # np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise


def construct_sparse_mat(indice, N):
    cur_row = -1
    new_indices = []
    new_indptr = []

    # for i, j in tqdm(indice):
    for i, j in indice:
        if i >= j:
            continue

        while i > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        new_indices.append(j)

    while N > cur_row:
        new_indptr.append(len(new_indices))
        cur_row += 1

    data = np.ones(len(new_indices), dtype=np.int64)
    indices = np.asarray(new_indices, dtype=np.int64)
    indptr = np.asarray(new_indptr, dtype=np.int64)

    mat = sp.csr_matrix((data, indices, indptr), (N, N))

    return mat + mat.T
