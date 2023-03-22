import random

import torch
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split


class TargetModel(torch.nn.Module):
    def __init__(self, dataset, mode, num_nodes, num_test):
        super(TargetModel, self).__init__()
        self.mode = mode
        self.num_nodes = num_nodes
        self.num_test = num_test
        self.conv1 = GCNConv(dataset.num_node_features, 256)
        self.conv2 = GCNConv(256, dataset.num_classes)

    def forward(self, x, edge_index):
        edges_raw = edge_index.cpu().numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
        G = nx.Graph()
        if self.mode == "TSTF":
            G.add_nodes_from(list(range(self.num_nodes)))
        else:
            G.add_nodes_from(list(range(self.num_test)))
        G.add_edges_from(edges)

        all_node_and_neighbors = []
        all_nodes = []

        for n in range(0, x.size(0)):
            all_nodes.append(n)
            all_node_and_neighbors.append((n, [node for node in G.neighbors(n)]))

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), all_node_and_neighbors


class ShadowModel(torch.nn.Module):
    def __init__(self, dataset, mode, num_nodes, num_test):
        super(ShadowModel, self).__init__()
        self.mode = mode
        self.num_nodes = num_nodes
        self.num_test = num_test
        self.conv1 = GCNConv(dataset.num_node_features, 256)
        self.conv2 = GCNConv(256, dataset.num_classes)

    def forward(self, x, edge_index):
        edges_raw = edge_index.cpu().numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
        G = nx.Graph()
        if self.mode == "TSTF":
            G.add_nodes_from(list(range(self.num_nodes)))
        else:
            G.add_nodes_from(list(range(self.num_test)))
        G.add_edges_from(edges)

        all_node_and_neighbors = []
        all_nodes = []

        for n in range(0, x.size(0)):
            all_nodes.append(n)
            all_node_and_neighbors.append((n, [node for node in G.neighbors(n)]))

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), all_node_and_neighbors


class AttackModel(torch.nn.Module):
    def __init__(self, dataset):
        super(AttackModel, self).__init__()
        self.fc1 = torch.nn.Linear(dataset.num_classes, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x)
        x = self.softmax(x)

        return x


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def mia_bb_func(dataset, attack_mode, device):
    num_train_Train_per_class = 90
    num_train_Shadow_per_class = 90
    num_test_Target = 630
    num_test_Shadow = 630

    data = dataset[0]
    label_list = [x for x in range(dataset.num_classes)]
    label_idx = data.y.numpy().tolist()

    data_new = get_inductive_split(attack_mode, data, dataset.num_classes, num_train_Train_per_class,
                                   num_train_Shadow_per_class, num_test_Target, num_test_Shadow)

    # bool_tensor = torch.ones(num_test_Target, dtype=torch.bool)
    #
    # if attack_mode == "TSTF":
    #     target_train_loader = NeighborSampler(data_new.target_edge_index, node_idx=bool_tensor, sizes=[25, 10],
    #                                           num_nodes=num_test_Target, batch_size=64, shuffle=False)
    #     shadow_train_loader = NeighborSampler(data_new.shadow_edge_index, node_idx=bool_tensor, sizes=[25, 10],
    #                                           num_nodes=num_test_Shadow, batch_size=64, shuffle=False)
    #     all_graph_loader = NeighborSampler(data_new.all_edge_index, node_idx=None, sizes=[-1],
    #                                        batch_size=1024, num_nodes=data.num_nodes, shuffle=False)
    # else:
    #     target_train_loader = NeighborSampler(data_new.target_edge_index, node_idx=bool_tensor, sizes=[25, 10],
    #                                           num_nodes=num_test_Target, batch_size=64, shuffle=False)
    #     target_test_loader = NeighborSampler(data_new.target_test_edge_index, node_idx=bool_tensor, sizes=[25, 10],
    #                                          num_nodes=num_test_Target, batch_size=64, shuffle=False)
    #     shadow_train_loader = NeighborSampler(data_new.shadow_edge_index, node_idx=bool_tensor, sizes=[25, 10],
    #                                           num_nodes=num_test_Shadow, batch_size=64, shuffle=False)
    #     shadow_test_loader = NeighborSampler(data_new.shadow_test_edge_index, node_idx=bool_tensor, sizes=[25, 10],
    #                                          num_nodes=num_test_Shadow, batch_size=64, shuffle=False)

    target_model = TargetModel(dataset, attack_mode, data.num_nodes, num_test_Target).to(device)
    shadow_model = ShadowModel(dataset, attack_mode, data.num_nodes, num_test_Target).to(device)
    data_new = data_new.to(device)

    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0001)
    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.0001)

    model_training_epoch = 301

    # target model training
    for epoch in range(1, model_training_epoch):
        approx_train_acc, train_loss = train_model(data_new, target_model, target_optimizer)
        train_acc, test_acc, macro, micro = test_model(attack_mode, num_test_Target, data_new, target_model,
                                                       label_list, device)
        log = 'TargetModel Epoch: {:03d}, Approx Train: {:.4f}, Train: {:.4f}, ' \
              'Test: {:.4f}, marco: {:.4f}, micro: {:.4f}'
        print(log.format(epoch, approx_train_acc, train_acc, test_acc, macro, micro))

    # shadow model training
    for epoch in range(1, model_training_epoch):
        approx_train_acc, train_loss = train_model(data_new, shadow_model, shadow_optimizer, isTarget=False)
        train_acc, test_acc, macro, micro = test_model(attack_mode, num_test_Target, data_new, shadow_model,
                                                       label_list, device, isTarget=False)
        log = 'ShadowModel Epoch: {:03d}, Approx Train: {:.4f}, Train: {:.4f}, ' \
              'Test: {:.4f}, marco: {:.4f}, micro: {:.4f}'
        print(log.format(epoch, approx_train_acc, train_acc, test_acc, macro, micro))

    # construct train & test set for attack model
    # attack training data
    attack_train_pos = pd.read_csv('mia_shadow_member_posteriors.txt', header=None, sep=' ')
    attack_train_pos["label"] = 1
    attack_train_neg = pd.read_csv('mia_shadow_non_member_posteriors.txt', header=None, sep=' ')
    attack_train_neg["label"] = 0

    attack_train_data = pd.concat([attack_train_pos, attack_train_neg])

    # attack testing data
    attack_test_pos = pd.read_csv('mia_target_member_posteriors.txt', header=None, sep=' ')
    attack_test_pos["label"] = 1
    attack_test_pos["nodeId"] = range(0, num_test_Target)

    attack_test_neg = pd.read_csv('mia_target_non_member_posteriors.txt', header=None, sep=' ')
    idxs = [int(i) for i in attack_test_neg.iloc[:, -1]]
    attack_test_neg = attack_test_neg.iloc[:, :-1]
    attack_test_neg["label"] = 0
    if attack_mode == "TSTF":
        attack_test_neg["nodeId"] = idxs
    else:
        attack_test_neg["nodeId"] = range(num_test_Target, num_test_Target + num_test_Target)

    attack_test_data = pd.concat([attack_test_pos, attack_test_neg])

    # all train
    X_attack_train = attack_train_data.drop("label", axis=1).values
    y_attack_train = attack_train_data["label"].values

    # all test
    X_attack_test = attack_test_data.drop(["nodeId", "label"], axis=1).values
    y_attack_test = attack_test_data["label"].values
    nodeId_test = attack_test_data["nodeId"].values

    # leave 50 training samples for evaluation
    attack_train_data_X, attack_test_data_X, attack_train_data_y, attack_test_data_y \
        = train_test_split(X_attack_train, y_attack_train, test_size=50, stratify=y_attack_train, random_state=42)
    attack_train_data = TensorDataset(torch.from_numpy(attack_train_data_X).float(), torch.from_numpy(
        attack_train_data_y))
    attack_train_data_loader = DataLoader(attack_train_data, batch_size=32, shuffle=True)
    attack_test_data = TensorDataset(torch.from_numpy(attack_test_data_X).float(), torch.from_numpy(
        attack_test_data_y))
    attack_test_data_loader = DataLoader(attack_test_data, batch_size=32, shuffle=True)

    # testing data
    test_data = TensorDataset(torch.from_numpy(X_attack_test).float(), torch.from_numpy(y_attack_test),
                              torch.from_numpy(nodeId_test))
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # training attack model
    attack_model = AttackModel(dataset).to(device)
    attack_model.apply(init_weights)
    criterion = torch.nn.CrossEntropyLoss()
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)
    epochs = 100

    attack_train(attack_model, device, attack_train_data_loader, attack_test_data_loader, criterion,
                 attack_optimizer, epochs)
    # test to confirm using attack_test_data_loader
    _, test_accuracyConfirmTest, posteriors, auroc, precision, \
    recall, f_score, _, _ = attack_test(device, criterion, attack_model, attack_test_data_loader, trainTest=True)

    # This is d result on the test set we used i.e split the attack data into train and test.
    # Size of the test = 50
    print("To confirm using attack_test_data_loader (50 test samples): {:.3f}".format(test_accuracyConfirmTest),
          "AUROC: {:.3f}".format(auroc), "precision: {:.3f}".format(precision), "recall {:.3f}".format(recall))

    # test for InOut train target data
    # This is the one we are interested in
    _, test_accuracyInOut, posteriors, auroc, precision, recall, f_score, \
    true_predicted_nodeIDs_and_class, false_predicted_nodeIDs_and_class = attack_test(device, criterion,
                                                                                      attack_model, test_data_loader)
    print("Test accuracy with Target Train InOut: {:.3f}".format(test_accuracyInOut), "AUROC: {:.3f}".format(auroc),
          "precision: {:.3f}".format(precision), "recall {:.3f}".format(recall), "F1 score {:.3f}".format(f_score),
          "===> Attack Performance!")


def get_inductive_split(mode, data, num_classes, num_train_Train_per_class,
                        num_train_Shadow_per_class, num_test_Target, num_test_Shadow):
    label_idx = data.y.numpy().tolist()
    print("label_idx", len(label_idx))
    target_train_idx = []
    shadow_train_idx = []

    for c in range(num_classes):
        idx = (data.y == c).nonzero().view(-1)
        sample_train_idx = idx[torch.randperm(idx.size(0))]
        sample_target_train_idx = sample_train_idx[:num_train_Train_per_class]
        sample_target_train_idx = [x.item() for x in sample_target_train_idx]
        target_train_idx.extend(sample_target_train_idx)

        sample_shadow_train_idx = sample_train_idx[
                                  num_train_Train_per_class:num_train_Train_per_class + num_train_Shadow_per_class]
        sample_shadow_train_idx = [x.item() for x in sample_shadow_train_idx]
        shadow_train_idx.extend(sample_shadow_train_idx)
    print("shadow train idx", len(shadow_train_idx))
    print("target train idx", len(target_train_idx))

    others = [x for x in range(len(label_idx)) if x not in set(target_train_idx) and x not in set(shadow_train_idx)]
    target_test_idx = random.sample(others, num_test_Target)
    shadow_test = [x for x in others if x not in set(target_test_idx)]
    shadow_test_idx = random.sample(shadow_test, num_test_Shadow)

    print("target_test_idx", len(target_test_idx))
    print("shadow_test_idx", len(shadow_test_idx))

    target_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    shadow_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for i in target_train_idx:
        target_train_mask[i] = True
    for i in shadow_train_idx:
        shadow_train_mask[i] = True

    target_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for i in target_test_idx:
        target_test_mask[i] = True
    shadow_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for i in shadow_test_idx:
        shadow_test_mask[i] = True

    target_x_inductive = data.x[target_train_idx]
    target_y_inductive = data.y[target_train_idx]
    target_edge_index_inductive, _ = subgraph(target_train_idx, data.edge_index)
    shadow_x_inductive = data.x[shadow_train_idx]
    shadow_y_inductive = data.y[shadow_train_idx]
    shadow_edge_index_inductive, _ = subgraph(shadow_train_idx, data.edge_index)

    if mode == 'TSTS':
        target_x_test_inductive = data.x[target_test_idx]
        target_y_test_inductive = data.y[target_test_idx]
        target_test_edge_index_inductive, _ = subgraph(target_test_idx, data.edge_index)

        shadow_x_test_inductive = data.x[shadow_test_idx]
        shadow_y_test_inductive = data.y[shadow_test_idx]
        shadow_test_edge_index_inductive, _ = subgraph(shadow_test_idx, data.edge_index)

    target_vertex_map = {}
    ind = -1
    for i in range(data.num_nodes):
        if i in target_train_idx:
            ind += 1
            target_vertex_map[i] = ind

    for i in range(target_edge_index_inductive.shape[1]):
        target_edge_index_inductive[0, i] = target_vertex_map[target_edge_index_inductive[0, i].tolist()]
        target_edge_index_inductive[1, i] = target_vertex_map[target_edge_index_inductive[1, i].tolist()]

    shadow_vertex_map = {}
    ind = -1
    for i in range(data.num_nodes):
        if i in shadow_train_idx:
            ind += 1
            shadow_vertex_map[i] = ind
    for i in range(shadow_edge_index_inductive.shape[1]):
        shadow_edge_index_inductive[0, i] = shadow_vertex_map[shadow_edge_index_inductive[0, i].tolist()]
        shadow_edge_index_inductive[1, i] = shadow_vertex_map[shadow_edge_index_inductive[1, i].tolist()]

    if mode == 'TSTS':
        target_test_vertex_map = {}
        ind = -1
        for i in range(data.num_nodes):
            if i in target_test_idx:
                ind += 1
                target_test_vertex_map[i] = ind
        for i in range(target_test_edge_index_inductive.shape[1]):
            target_test_edge_index_inductive[0, i] = target_test_vertex_map[target_test_edge_index_inductive[0, i].tolist()]
            target_test_edge_index_inductive[1, i] = target_test_vertex_map[target_test_edge_index_inductive[1, i].tolist()]

        shadow_test_vertex_map = {}
        ind = -1
        for i in range(data.num_nodes):
            if i in shadow_test_idx:
                ind += 1
                shadow_test_vertex_map[i] = ind
        for i in range(shadow_test_edge_index_inductive.shape[1]):
            shadow_test_edge_index_inductive[0, i] = shadow_test_vertex_map[shadow_test_edge_index_inductive[0, i].tolist()]
            shadow_test_edge_index_inductive[1, i] = shadow_test_vertex_map[shadow_test_edge_index_inductive[1, i].tolist()]

    all_x = data.x
    all_y = data.y
    all_edge_index = data.edge_index

    if mode == 'TSTS':
        data = Data(target_x=target_x_inductive, target_edge_index=target_edge_index_inductive,
                    target_y=target_y_inductive, target_test_x=target_x_test_inductive,
                    target_test_edge_index=target_test_edge_index_inductive, target_test_y=target_y_test_inductive,
                    shadow_x=shadow_x_inductive, shadow_edge_index=shadow_edge_index_inductive,
                    shadow_y=shadow_y_inductive, shadow_test_x=shadow_x_test_inductive,
                    shadow_test_edge_index=shadow_test_edge_index_inductive, shadow_test_y=shadow_y_test_inductive,
                    target_train_mask=target_train_mask, shadow_train_mask=shadow_train_mask, all_x=all_x,
                    all_edge_index=all_edge_index, all_y=all_y, target_test_mask=target_test_mask,
                    shadow_test_mask=shadow_test_mask)
    else:
        data = Data(target_x=target_x_inductive, target_edge_index=target_edge_index_inductive, target_y=target_y_inductive,
                    shadow_x=shadow_x_inductive, shadow_edge_index=shadow_edge_index_inductive, shadow_y=shadow_y_inductive,
                    target_train_mask=target_train_mask, shadow_train_mask=shadow_train_mask, all_x=all_x, all_y=all_y,
                    all_edge_index=all_edge_index, target_test_mask=target_test_mask, shadow_test_mask=shadow_test_mask)

    return data


def train_model(data, model, optimizer, isTarget=True):
    model.train()
    optimizer.zero_grad()
    if isTarget:
        out, nodes_and_neighbors = model(data.target_x,
                                         data.target_edge_index)
        loss = F.nll_loss(out, data.target_y)
    else:
        out, nodes_and_neighbors = model(data.shadow_x, data.shadow_edge_index)
        loss = F.nll_loss(out, data.shadow_y)

    pred = torch.exp(out)

    loss.backward()
    optimizer.step()

    # approximate accuracy
    if isTarget:
        train_loss = loss.item() / int(data.target_train_mask.sum())
        total_correct = int(pred.argmax(dim=-1).eq(data.target_y).sum()) / int(data.target_train_mask.sum())
    else:
        train_loss = loss.item() / int(data.shadow_train_mask.sum())
        total_correct = int(pred.argmax(dim=-1).eq(data.shadow_y).sum()) / int(data.shadow_train_mask.sum())

    return total_correct, train_loss


def test_model(mode, num_test_Target, data, model, label_list, device, isTarget=True):
    save_target_InTrain = 'mia_target_member_posteriors.txt'
    save_target_OutTrain = 'mia_target_non_member_posteriors.txt'
    save_target_InTrain_nodes_neigbors = 'mia_target_member_nodes_neighbors.npy'
    save_target_OutTrain_nodes_neigbors = 'mia_target_non_member_nodes_neighbors.npy'
    save_shadow_InTrain = 'mia_shadow_member_posteriors.txt'
    save_shadow_OutTrain = 'mia_shadow_non_member_posteriors.txt'
    save_shadow_InTrain_nodes_neigbors = 'mia_shadow_member_nodes_neighbors.npy'
    save_shadow_OutTrain_nodes_neigbors = 'mia_shadow_non_member_nodes_neighbors.npy'

    model.eval()
    # Also changed this to give true test using full graph. This will give the true train result--No it wont.
    # See comment below
    # This is a better n accurate approach
    if isTarget:
        '''InTrain Target'''
        pred, nodes_and_neighbors = model(data.target_x, data.target_edge_index)
        pred_Intrain = pred.max(1)[1].to(device)
        # Actual probabilities
        pred_Intrain_ps = torch.exp(pred)
        np.savetxt(save_target_InTrain, pred_Intrain_ps.cpu().detach().numpy())
        nodes_and_neighbors = np.array(nodes_and_neighbors, dtype=object)
        np.save(save_target_InTrain_nodes_neigbors, nodes_and_neighbors)

        '''OutTrain Target'''
        if mode == 'TSTF':
            preds, nodes_and_neighbors = model(data.all_x, data.all_edge_index)
            nodes_and_neighbors = np.array(nodes_and_neighbors, dtype=object)
            preds = preds[data.target_test_mask]

            mask = data.target_test_mask.gt(0)
            mask = mask.cpu().numpy()

            nodes_and_neighbors = nodes_and_neighbors[mask]

            pred_out = preds.max(1)[1].to(device)
            pred_out_ps = torch.exp(preds)

            incremented_nodes_and_neighbors = []

            for i in range(len(nodes_and_neighbors)):
                res = nodes_and_neighbors[i][1]
                res_0 = nodes_and_neighbors[i][0]
                incremented_nodes_and_neighbors.append((res_0, res))

            incremented_nodes_and_neighbors = np.array(incremented_nodes_and_neighbors, dtype=object)
            np.save(save_target_OutTrain_nodes_neigbors, incremented_nodes_and_neighbors)

            nodes = []
            for i in range(0, len(incremented_nodes_and_neighbors)):
                nodes.append(incremented_nodes_and_neighbors[i][0])
            nodes = np.array(nodes)
            preds_and_nodes = np.column_stack((pred_out_ps.cpu().detach().numpy(), nodes))
            np.savetxt(save_target_OutTrain, preds_and_nodes)

            pred_labels = pred_out.tolist()
            true_labels = data.all_y[data.target_test_mask].tolist()

            # The train accuracy is not on the full graph. It's similar to approx_train_acc
            train_acc = get_train_acc(data, pred_Intrain)
            # Test n val are on full graph
            test_acc = get_test_acc(data, pred_out)
        else:
            preds, nodes_and_neighbors = model(data.target_test_x, data.target_test_edge_index)
            pred_out = pred.max(1)[1].to(device)
            pred_out_ps = torch.exp(preds)
            np.savetxt(save_target_OutTrain, pred_out_ps.cpu().detach().numpy())
            incremented_nodes_and_neighbors = []
            for i in range(len(nodes_and_neighbors)):
                res = [x + num_test_Target for x in nodes_and_neighbors[i][1]]
                res_0 = nodes_and_neighbors[i][0] + num_test_Target
                incremented_nodes_and_neighbors.append((res_0, res))

            incremented_nodes_and_neighbors = np.array(incremented_nodes_and_neighbors, dtype=object)
            np.save(save_target_OutTrain_nodes_neigbors, incremented_nodes_and_neighbors)
            pred_labels = pred_out.tolist()
            true_labels = data.target_test_y.tolist()
            train_acc = get_train_acc(data, pred_Intrain)
            test_acc = get_test_acc(data, pred_out)

    else:

        '''InTrain Shadow'''
        pred, nodes_and_neighbors = model(data.shadow_x, data.shadow_edge_index)
        pred_Intrain = pred.max(1)[1].to(device)
        # Actual probabilities
        pred_Intrain_ps = torch.exp(pred)
        np.savetxt(save_shadow_InTrain, pred_Intrain_ps.cpu().detach().numpy())

        '''OutTrain Shadow'''
        if mode == 'TSTF':
            preds, nodes_and_neighbors = model(data.all_x, data.all_edge_index)
            nodes_and_neighbors = np.array(nodes_and_neighbors, dtype=object)

            preds = preds[data.shadow_test_mask]

            mask = data.shadow_test_mask.gt(0)  # trick to
            mask = mask.cpu().numpy()

            nodes_and_neighbors = nodes_and_neighbors[mask]

            pred_out = preds.max(1)[1].to(device)
            pred_out_ps = torch.exp(preds)
            np.savetxt(save_shadow_OutTrain, pred_out_ps.cpu().detach().numpy())

            pred_labels = pred_out.tolist()
            true_labels = data.all_y[data.shadow_test_mask].tolist()

            # The train accuracy is not on the full graph. It's similar to approx_train_acc
            train_acc = get_train_acc(data, pred_Intrain, False)
            # Test n val are on full graph
            test_acc = get_test_acc(data, pred_out, False)
        else:
            preds, nodes_and_neighbors = model(data.shadow_test_x, data.shadow_test_edge_index)
            pred_out = preds.max(1)[1].to(device)
            pred_out_ps = torch.exp(preds)
            np.savetxt(save_shadow_OutTrain, pred_out_ps.cpu().detach().numpy())

            pred_labels = pred_out.tolist()
            true_labels = data.shadow_test_y.tolist()

            # The train accuracy is not on the full graph. It's similar to approx_train_acc
            train_acc = get_train_acc(data, pred_Intrain, False)
            # Test n val are on full graph
            test_acc = get_test_acc(data, pred_out, False)

    # pred_Intrain = model(data_new.all_x,data_new.all_edge_index)[data_new.target_train_mask].max(1)[1]
    # # Actual probabilities
    # pred_Intrain_ps = torch.exp(model(data_new.all_x,data_new.all_edge_index)[data_new.target_train_mask])

    # print("posteriors", pred_Intrain)

    # The f1 measures are on test dataset

    f1_marco = get_marco_f1(data, pred_labels, true_labels, label_list)
    f1_micro = get_micro_f1(data, pred_labels, true_labels, label_list)

    return train_acc, test_acc, f1_marco, f1_micro


def get_train_acc(data, pred, isTarget=True):
    if isTarget:
        # Removed train mask cos u r testing on the subgraph only tho
        train_acc = pred.eq(data.target_y).sum().item() / data.target_train_mask.sum().item()
    else:
        train_acc = pred.eq(data.shadow_y).sum().item() / data.shadow_train_mask.sum().item()

    return train_acc


def get_test_acc(data, pred, isTarget=True):
    if isTarget:
        test_acc = pred.eq(
            data.all_y[data.target_test_mask]).sum().item() / data.target_test_mask.sum().item()
    else:
        test_acc = pred.eq(
            data.all_y[data.shadow_test_mask]).sum().item() / data.shadow_test_mask.sum().item()
    return test_acc


def get_marco_f1(data_new, pred_labels, true_labels, label_list):
    # f1_marco = f1_score(true_labels,pred_labels,label_list,average='macro')
    f1_marco = f1_score(true_labels, pred_labels, average='macro')
    return f1_marco


def get_micro_f1(data_new, pred_labels, true_labels, label_list):
    # f1_micro = f1_score(true_labels,pred_labels,label_list,average='micro')
    f1_micro = f1_score(true_labels, pred_labels, average='micro')
    return f1_micro


def attack_train(model, device, trainloader, testloader, criterion, optimizer, epochs, steps=0):
    # train ntwk

    # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    final_train_loss = 0
    train_losses, test_losses = [], []
    posteriors = []
    for e in range(epochs):
        running_loss = 0
        train_accuracy = 0

        # This is features, labels cos we dont care about nodeID during training! only during test
        for features, labels in trainloader:
            model.train()
            features, labels = features.to(device), labels.to(device)

            # print("post shape", features.shape)
            # print("labels",labels)
            optimizer.zero_grad()
            # print("features", features.shape)

            # features = features.unsqueeze(1) #unsqueeze
            # flatten features
            features = features.view(features.shape[0], -1)

            logps = model(features)  # log probabilities
            # print("labelsssss", labels.shape)
            loss = criterion(logps, labels)

            # Actual probabilities
            ps = logps  # torch.exp(logps) #Only use this if the loss is nlloss
            # print("ppppp",ps)

            # top_p gives the probabilities while top_class gives the predicted classes
            top_p, top_class = ps.topk(1, dim=1)
            # print(top_p)
            equals = top_class == labels.view(
                *top_class.shape)  # making the shape of the label and top class the same
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            # Everything in this else block executes after every epock
            # print(f"training loss: {running_loss}")

            # test_loss = 0
            # test_accuracy = 0
            #
            # # Turn off gradients for validation, saves memory and computations
            # with torch.no_grad():
            #     # Doing validation
            #
            #     # set model to evaluation mode
            #     model.eval()
            #
            #     if e == epochs - 1:
            #         print("Doing attack validation===========")
            #     # validation pass
            #
            #     for features, labels in testloader:
            #         # features = features.unsqueeze(1)  # unsqueeze
            #         features = features.view(features.shape[0], -1)
            #         logps = model(features)
            #         test_loss += criterion(logps, labels)
            #
            #         # Actual probabilities
            #         ps = torch.exp(logps)
            ## top_p gives the probabilities while top_class gives the predicted classes
            #         top_p, top_class = ps.topk(1, dim=1)
            #         # print(top_p)
            #         equals = top_class == labels.view(
            #             *top_class.shape)  # making the shape of the label and top class the same
            #         test_accuracy += torch.mean(equals.type(torch.FloatTensor))
            test_loss, test_accuracy, _, _, _, _, _, _, _ = attack_test(device, criterion, model, testloader, trainTest=True)

            # set model back yo train model
            model.train()
            # scheduler.step()

            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss)

            # get final train loss. To be returned at the end of the training loop
            final_train_loss = running_loss / len(trainloader)

            print("Epoch: {}/{}..".format(e + 1, epochs),
                  "Training loss: {:.5f}..".format(running_loss / len(trainloader)),
                  "Test Loss: {:.5f}..".format(test_loss),
                  "Train Accuracy: {:.3f}".format(train_accuracy / len(trainloader)),
                  "Test Accuracy: {:.3f}".format(test_accuracy)
                  )
    return final_train_loss


def attack_test(device, criterion, model, testloader, singleClass=False, trainTest=False):
    test_loss = 0
    test_accuracy = 0
    auroc = 0
    precision = 0
    recall = 0
    f_score = 0

    posteriors = []
    all_nodeIDs = []
    true_predicted_nodeIDs_and_class = {}
    false_predicted_nodeIDs_and_class = {}

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        # Doing validation

        # set model to evaluation mode
        model.eval()

        if trainTest:
            for features, labels in testloader:
                features, labels = features.to(device), labels.to(device)
                # features = features.unsqueeze(1)  # unsqueeze
                features = features.view(features.shape[0], -1)
                logps = model(features)
                test_loss += criterion(logps, labels)

                # Actual probabilities
                ps = logps  # torch.exp(logps)
                posteriors.append(ps)

                # if singleclass=false
                if not singleClass:
                    y_true = labels.cpu().unsqueeze(-1)
                    # print("y_true", y_true)
                    y_pred = ps.argmax(dim=-1, keepdim=True)
                    # print("y_pred", y_pred)

                    # uncomment this to show AUROC
                    auroc += roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                    # print("auroc", auroc)

                    precision += precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    recall += recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    f_score += f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                # top_p gives the probabilities while top_class gives the predicted classes
                top_p, top_class = ps.topk(1, dim=1)
                # print(top_p)
                equals = top_class == labels.view(
                    *top_class.shape)  # making the shape of the label and top class the same
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        else:
            # print("len(testloader.dataset)", len(testloader.dataset))
            for features, labels, nodeIDs in testloader:
                # print("nodeIDs", nodeIDs)
                features, labels = features.to(device), labels.to(device)
                # features = features.unsqueeze(1)  # unsqueeze
                features = features.view(features.shape[0], -1)
                logps = model(features)
                test_loss += criterion(logps, labels)

                # Actual probabilities
                ps = logps  # torch.exp(logps)
                posteriors.append(ps)

                # print("ps", ps)
                # print("nodeIDs", nodeIDs)

                all_nodeIDs.append(nodeIDs)

                # if singleclass=false
                if not singleClass:
                    y_true = labels.cpu().unsqueeze(-1)
                    # print("y_true", y_true)
                    y_pred = ps.argmax(dim=-1, keepdim=True)
                    # print("y_pred", y_pred)

                    # uncomment this to show AUROC
                    auroc += roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
                    # print("auroc", auroc)

                    precision += precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    recall += recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                    f_score += f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                # top_p gives the probabilities while top_class gives the predicted classes
                top_p, top_class = ps.topk(1, dim=1)
                # print("top_p", top_p)
                # print("top_class", top_class)

                equals = top_class == labels.view(
                    *top_class.shape)  # making the shape of the label and top class the same

                # print("equals", len(equals))
                for i in range(len(equals)):
                    if equals[i]:  # if element is true {meaning both member n non-member}, get the nodeID
                        # print("baba")
                        # print("true pred nodeIDs", nodeIDs[i])
                        true_predicted_nodeIDs_and_class[nodeIDs[i].item()] = top_class[i].item()
                        # print("len(true_predicted_nodeIDs_and_class)", len(true_predicted_nodeIDs_and_class),
                        # "nodeID--",nodeIDs[i].item(), "class--",  top_class[i].item())
                    else:
                        false_predicted_nodeIDs_and_class[nodeIDs[i].item()] = top_class[i].item()
                        # print("len(false_predicted_nodeIDs_and_class)", len(false_predicted_nodeIDs_and_class))

                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

    test_accuracy = test_accuracy / len(testloader)
    test_loss = test_loss / len(testloader)
    final_auroc = auroc / len(testloader)
    final_precision = precision / len(testloader)
    final_recall = recall / len(testloader)
    final_f_score = f_score / len(testloader)

    return test_loss, test_accuracy, posteriors, final_auroc, final_precision, final_recall, final_f_score, \
           true_predicted_nodeIDs_and_class, false_predicted_nodeIDs_and_class
