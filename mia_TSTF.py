import torch
import networkx as nx
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborSampler
from sklearn.model_selection import train_test_split

from mia_utils import get_inductive_split, train_model, test_model, attack_train, attack_test


runs = 2


# params
target_model_name = "GCN"
shadow_model_name = "GCN"
dataset_name = "Cora"
attack_mode = "TSTF"


class TargetModel(torch.nn.Module):
    def __init__(self, dataset):
        super(TargetModel, self).__init__()
        if target_model_name == "GCN":
            self.conv1 = GCNConv(dataset.num_node_features, 256)
            self.conv2 = GCNConv(256, dataset.num_classes)
        else:
            raise NotImplementedError("Not implemented model.")

    def forward(self, x, edge_index):
        edges_raw = edge_index.cpu().numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
        G = nx.Graph()
        G.add_nodes_from(list(range(data.num_nodes)))
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
    def __init__(self, dataset):
        super(ShadowModel, self).__init__()
        if shadow_model_name == "GCN":
            self.conv1 = GCNConv(dataset.num_node_features, 256)
            self.conv2 = GCNConv(256, dataset.num_classes)
        else:
            raise NotImplementedError("Not implemented model.")

    def forward(self, x, edge_index):
        edges_raw = edge_index.cpu().numpy()
        edges = [(x, y) for x, y in zip(edges_raw[0, :], edges_raw[1, :])]
        G = nx.Graph()
        G.add_nodes_from(list(range(data.num_nodes)))
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
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


for which_run in range(1, runs):
    dataset = Planetoid('datasets', dataset_name, split='random')
    num_train_Train_per_class = 90
    num_train_Shadow_per_class = 90
    num_test_Target = 630
    num_test_Shadow = 630

    data = dataset[0]
    label_list = [x for x in range(dataset.num_classes)]
    label_idx = data.y.numpy().tolist()

    data_new = get_inductive_split(attack_mode, data, dataset.num_classes, num_train_Train_per_class, num_train_Shadow_per_class,
                                   num_test_Target, num_test_Shadow)

    bool_tensor = torch.ones(num_test_Target, dtype=torch.bool)
    # target_train_idx = torch.nonzero(data_new.target_train_mask).view(-1)
    # shadow_train_idx = torch.nonzero(data_new.shadow_train_mask).view(-1)

    target_train_loader = NeighborSampler(data_new.target_edge_index, node_idx=bool_tensor, sizes=[25, 10],
                                          num_nodes=num_test_Target, batch_size=64, shuffle=False)
    shadow_train_loader = NeighborSampler(data_new.shadow_edge_index, node_idx=bool_tensor, sizes=[25, 10],
                                          num_nodes=num_test_Shadow, batch_size=64, shuffle=False)
    all_graph_loader = NeighborSampler(data_new.all_edge_index, node_idx=None, sizes=[-1],
                                       batch_size=1024, num_nodes=data.num_nodes, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_model = TargetModel(dataset).to(device)
    shadow_model = ShadowModel(dataset).to(device)
    data_new = data_new.to(device)

    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0001)
    shadow_optimizer = torch.optim.Adam(shadow_model.parameters(), lr=0.0001)

    model_training_epoch = 301
    # target model training
    # *******************************************************
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
    attack_test_neg["nodeId"] = idxs
    attack_test_neg["label"] = 0

    attack_test_data = pd.concat([attack_test_pos, attack_test_neg])

    # all train
    X_attack_train = attack_train_data.drop("label", axis=1).values
    y_attack_train = attack_train_data["label"].values
    # positive train
    X_attack_train_pos = attack_train_pos.drop("label", axis=1).values
    y_attack_train_pos = attack_train_pos["label"].values
    # negative train
    X_attack_train_neg = attack_train_neg.drop("label", axis=1).values
    y_attack_train_neg = attack_train_neg["label"].values
    # all test
    X_attack_test = attack_test_data.drop(["nodeId", "label"], axis=1).values
    y_attack_test = attack_test_data["label"].values
    nodeId_test = attack_test_data["nodeId"].values
    # positive test
    X_attack_test_pos = attack_test_pos.drop(["nodeId", "label"], axis=1).values
    y_attack_test_pos = attack_test_pos["label"].values
    # negative test
    X_attack_test_neg = attack_test_neg.drop(["nodeId", "label"], axis=1).values
    y_attack_test_neg = attack_test_neg["label"].values

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
