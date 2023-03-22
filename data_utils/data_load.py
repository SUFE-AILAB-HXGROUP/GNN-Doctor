import logging

from torch_geometric import datasets
import torch_geometric.transforms as T


def load_dataset(dataset_name, setting, attack_name):
    """
    :param dataset_name: name of graph dataset
    :param setting: attack setting - black_box or node_emb_publish
    :param attack_name: name of attack method
    :return: dataset
    """

    if setting == 'node_emb_publish':
        if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
            transform = T.Compose([
                T.NormalizeFeatures(),
                T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=False),
            ])
            dataset = datasets.Planetoid('datasets', dataset_name, transform=transform)

            return dataset
    elif setting == 'black_box':
        if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
            transform = T.Compose([T.NormalizeFeatures()])
            dataset = datasets.Planetoid('datasets', dataset_name, transform=transform)

            return dataset

    else:
        logging.error("Not implemented setting. Please set setting=='node_emb_publish' or 'black_box'.")

        return None