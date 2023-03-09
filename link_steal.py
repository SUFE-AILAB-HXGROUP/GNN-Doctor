import json
import time
import random

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine, euclidean, correlation, chebyshev,\
    braycurtis, canberra, cityblock, sqeuclidean


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, out_channels=16, cached=True)
        self.conv2 = GCNConv(in_channels=16, out_channels=dataset.num_classes, cached=True)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = Linear(in_features=dataset.num_features, out_features=16)
        self.dense2 = Linear(in_features=16, out_features=dataset.num_classes)

    def forward(self):
        x = data.x
        x = self.dense1(x).relu()
        x = F.dropout(x, training=self.training)
        x = self.dense2(x).relu()
        x = F.dropout(x, training=self.training)
        return F.log_softmax(x, dim=1)


def train_model():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test_model():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


@torch.no_grad()
def test_all():
    model.eval()
    logits, accs = model(), []
    return logits


def train_mlp():
    mlp.train()
    mlp_optimizer.zero_grad()
    F.nll_loss(mlp()[data.train_mask], data.y[data.train_mask]).backward()
    mlp_optimizer.step()


@torch.no_grad()
def test_mlp():
    mlp.eval()
    logits, accs = mlp(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


@torch.no_grad()
def test_all_mlp():
    mlp.eval()
    logits, accs = mlp(), []
    return logits


def get_link(adj, node_num):
    unlink = []
    link = []
    existing_set = set([])
    rows, cols = adj.nonzero()
    print("There are %d edges in this dataset" % len(rows))
    for i in range(len(rows)):
        r_index = rows[i]
        c_index = cols[i]
        if r_index < c_index:
            link.append([r_index, c_index])
            existing_set.add(",".join([str(r_index), str(c_index)]))

    random.seed(1)
    t_start = time.time()
    while len(unlink) < len(link):
        if len(unlink) % 1000 == 0:
            print(len(unlink), time.time() - t_start)

        row = random.randint(0, node_num - 1)
        col = random.randint(0, node_num - 1)
        if row > col:
            row, col = col, row
        edge_str = ",".join([str(row), str(col)])
        if (row != col) and (edge_str not in existing_set):
            unlink.append([row, col])
            existing_set.add(edge_str)
    return link, unlink


def generate_train_test(link, unlink, dense_pred, gcn_pred, train_ratio):
    train = []
    test = []

    train_len = len(link) * train_ratio
    for i in range(len(link)):
        # print(i)
        link_id0 = link[i][0]
        link_id1 = link[i][1]

        line_link = {
            'label': 1,
            'gcn_pred0': gcn_pred[link_id0],
            'gcn_pred1': gcn_pred[link_id1],
            "dense_pred0": dense_pred[link_id0],
            "dense_pred1": dense_pred[link_id1],
            "feature_arr0": feature_arr[link_id0],
            "feature_arr1": feature_arr[link_id1],
            "id_pair":[int(link_id0),int(link_id1)]
        }

        unlink_id0 = unlink[i][0]
        unlink_id1 = unlink[i][1]

        line_unlink = {
            'label': 0,
            'gcn_pred0': gcn_pred[unlink_id0],
            'gcn_pred1': gcn_pred[unlink_id1],
            "dense_pred0": dense_pred[unlink_id0],
            "dense_pred1": dense_pred[unlink_id1],
            "feature_arr0": feature_arr[unlink_id0],
            "feature_arr1": feature_arr[unlink_id1],
            "id_pair":[int(unlink_id0),int(unlink_id1)]
        }

        if i < train_len:
            train.append(line_link)
            train.append(line_unlink)
        else:
            test.append(line_link)
            test.append(line_unlink)

    with open(
             "link_steal_tmp/%s_train_ratio_%0.1f_train.json" %
        (dataset_name, train_ratio), "w") as wf1, open(
            "link_steal_tmp/%s_train_ratio_%0.1f_test.json" %
            (dataset_name, train_ratio), "w") as wf2:
        for row in train:
            wf1.write("%s\n" % json.dumps(row))
        for row in test:
            wf2.write("%s\n" % json.dumps(row))
    return train, test


def attack_0(target_posterior_list):
    sim_metric_list = [cosine, euclidean, correlation, chebyshev,
                       braycurtis, canberra, cityblock, sqeuclidean]
    sim_list_target = [[] for _ in range(len(sim_metric_list))]
    for i in range(len(target_posterior_list)):
        for j in range(len(sim_metric_list)):
            # using target only
            target_sim = sim_metric_list[j](target_posterior_list[i][0],
                                            target_posterior_list[i][1])
            sim_list_target[j].append(target_sim)
    return sim_list_target


def write_auc(pred_prob_list, label, desc):
    print("Attack 0 " + desc)
    sim_list_str = ['cosine', 'euclidean', 'correlation', 'chebyshev',
                    'braycurtis', 'canberra', 'cityblock', 'sqeuclidean']
    with open("link_steal_tmp/attack_0.txt", "a") as wf:
        for i in range(len(sim_list_str)):
            pred = np.array(pred_prob_list[i], dtype=np.float64)
            where_are_nan = np.isnan(pred)
            where_are_inf = np.isinf(pred)
            pred[where_are_nan] = 0
            pred[where_are_inf] = 0

            i_auc = roc_auc_score(label, pred)
            if i_auc < 0.5:
                i_auc = 1 - i_auc
            print(sim_list_str[i], i_auc)
            wf.write(
                "%s,%s,%d,%0.5f,%s\n" %
                (dataset_name, "attack0_%s_%s" %
                 (desc, sim_list_str[i]), -1, i_auc, 0.5))


if __name__ == '__main__':
    dataset_name = 'Cora'
    path = 'datasets'
    dataset = Planetoid(root=path, name=dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    mlp = MLP().to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)
    mlp_optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)

    # train target model
    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train_model()
        train_acc, val_acc, tmp_test_acc = test_model()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train_mlp()
        train_acc, val_acc, tmp_test_acc = test_mlp()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    gcn_pred = test_all().cpu().detach().numpy().tolist()
    mlp_pred = test_all_mlp().cpu().detach().numpy().tolist()

    # start attack
    feature_arr = data.x.cpu().numpy().tolist()

    adj = to_dense_adj(data.edge_index.cpu())[0].numpy()
    node_num = adj.shape[0]
    link, unlink = get_link(adj, node_num)

    random.shuffle(link)
    random.shuffle(unlink)
    edge_label = []
    for row in link:
        edge_label.append(1)
    for row in unlink:
        edge_label.append(0)

    for i in range(1, 11):
        print("generating: %d percent" % (i * 10))
        generate_train_test(link, unlink, mlp_pred, gcn_pred, i / 10.0)

    test_path = "link_steal_tmp/%s_train_ratio_%s_test.json" % (dataset_name, "0.9")
    test_data = open(test_path).readlines()  # read test data only
    label_list = []
    target_posterior_list = []
    reference_posterior_list = []
    feature_list = []
    for row in test_data:
        row = json.loads(row)
        label_list.append(row["label"])
        target_posterior_list.append([row["gcn_pred0"], row["gcn_pred1"]])
        reference_posterior_list.append(
            [row["dense_pred0"], row["dense_pred1"]])
        feature_list.append([row["feature_arr0"], row["feature_arr1"]])

    sim_list_target = attack_0(target_posterior_list)
    write_auc(sim_list_target, label_list, desc="target posterior similarity")

