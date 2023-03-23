import random

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data


def pokec_process(dataset_name):
    if dataset_name == 'pokec_z':
        idx_features_labels = pd.read_csv('datasets/pokec/region_job.csv')
        edges_unordered = np.genfromtxt("datasets/pokec/region_job_relationship.txt", dtype=int)
    else:
        idx_features_labels = pd.read_csv('datasets/pokec/region_job_2.csv')
        edges_unordered = np.genfromtxt("datasets/pokec/region_job_2_relationship.txt", dtype=int)

    sens_attr = "region"
    predict_attr = "I_am_working_in_field"

    header = list(idx_features_labels.columns)
    header.remove("user_id")
    header.remove(sens_attr)
    header.remove(predict_attr)

    features = idx_features_labels[header].values
    labels = idx_features_labels[predict_attr].values
    sensitive_attr = idx_features_labels[sens_attr].values

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int64).reshape(
        edges_unordered.shape).T

    edge_index = torch.from_numpy(edges).to(torch.int64)
    x = torch.from_numpy(features).to(torch.float32)
    y = torch.from_numpy(labels).to(torch.int64)
    sensitive_attr = torch.from_numpy(sensitive_attr).to(torch.int64)

    random.seed(42)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)
    idx_train = label_idx[:int(0.5 * len(label_idx))]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    idx_test = label_idx[int(0.75 * len(label_idx)):]

    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    data = Data(x=x, edge_index=edge_index, edge_attr=None, y=y, priv_label=sensitive_attr)

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    dataset = [data]

    return dataset
