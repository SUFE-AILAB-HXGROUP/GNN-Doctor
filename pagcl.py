import argparse

import torch
from torch_geometric.utils import to_dense_adj, to_undirected

from data_utils import load_dataset


def drop_feature_priv(x):
    drop_prob = torch.zeros_like(x)
    for i in range(test_edge_true2.shape[1]):
        drop_prob[test_edge_true2[random.choice([0, 1]), i],
                  x[test_edge_true2[0, i], :] == x[test_edge_true2[1, i], :]] = args.ratio
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[drop_mask] = 1 - x[drop_mask]
    return x


def drop_feature_priv2(x):
    drop_prob = torch.zeros_like(x)
    for i in range(test_edge_true2.shape[1]):
        drop_prob[test_edge_true2[random.choice([0, 1]), i],
                  x[test_edge_true2[0, i], :] == x[test_edge_true2[1, i], :]] = args.ratio
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[drop_mask] = 0.
    return x


def get_priv_adj(test_data, num_nodes):
    """
    generate an adjacency matrix indicates private neighbors
    param:
        test_data: test data splited by PyG
        num_nodes: number of nodes in original graph
    return:
        priv_adj: matrix indicates private neighbors
    """
    priv_edge_num = int(len(test_data.edge_label) / 2)
    edge_index = test_data.edge_label_index[:, :priv_edge_num]
    priv_adj = to_dense_adj(to_undirected(edge_index), max_num_nodes=num_nodes)[0]

    return priv_adj


def get_weight_matrix(priv_adj, weight=1):
    ones_matrix = torch.ones_like(priv_adj)
    w_mat = ones_matrix * weight
    weight_matrix = torch.where(priv_adj == 1, w_mat, ones_matrix)

    return weight_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--topk', type=str, default=1)
    parser.add_argument('--weight', type=float, default=3.0)

    args = parser.parse_args()

    if args.device == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    dataset = load_dataset(args.dataset)
    train_data, val_data, test_data = dataset[0]

    topk = args.topk

    num_nodes = train_data.num_nodes
    priv_neighbor_adj = get_priv_adj(test_data, num_nodes)

    #
    weights = args.weight
    weights_matrix = get_weight_matrix(priv_neighbor_adj, weights).to(device)

    encoder = Encoder(dataset.num_features, )
    model = GRACE_priv()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    # edge drop weights
    degree_weights = degree_drop_weights(data.edge_index).to(device)
    if param['drop_scheme'] == 'pagcl_v1':
        sp_masks = pagcl_weights(degree_weights, data, test_data, k=topk)
    elif param['drop_scheme'] == 'pagcl_v2':
        sp_masks = pagcl_weights_v2(degree_weights, data, test_data, k=3)


    # model training
    for epoch in range(1, param['num_epochs']+1):
        edge_index_1 = drop_edge_weighted_combine(sp_masks, data.edge_index, degree_weights,
                                                  p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        edge_index_2 = drop_edge_weighted_combine(sp_masks, data.edge_index, degree_weights,
                                                  p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        if args.feature == 'v1':
            x_1 = drop_feature_priv(x_1)
            x_2 = drop_feature_priv(x_2)
        elif args.feature == 'v2':
            x_1 = drop_feature_priv2(x_1)
            x_2 = drop_feature_priv2(x_2)
        else:
            x_1 = data.x
            x_2 = data.x

        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.loss_priv(z1, z2, weights_matrix)
        loss.backward()
        optimizer.step()
        print(loss.item())
