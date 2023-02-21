import numpy as np

from torch_geometric import datasets
import torch_geometric.transforms as T


def load_dataset(dataset_name):
    if dataset_name in ['Cora', 'Citeseer', 'PubMed']:
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True, add_negative_train_samples=False),
        ])
        dataset = datasets.Planetoid('datasets', dataset_name, transform=transform)

        return dataset
