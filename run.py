import argparse
import random

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

from data_utils import load_dataset
from gnns import GCN, UnsupGCN
from attack_utils import sim_attacks, ml_attacks
from attack_models import MulticlassEvaluator, log_regression


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    out = model.decode(z, edge_label_index).view(-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attack', type=str, default='link_infer')
    parser.add_argument('--target-model', type=str, default='gcn')
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)

    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = load_dataset(args.dataset)
    train_data, val_data, test_data = dataset[0]

    # model = GCN(input_dim=dataset.num_features, hidden_dim=args.hidden_dim, output_dim=dataset.num_classes)
    model = UnsupGCN(input_dim=dataset.num_features, hidden_dim=128, output_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Target model training
    best_val_auc = final_test_auc = 0
    for epoch in range(1, args.epoch + 1):
        loss = train()
        val_auc = test(val_data)
        test_auc = test(test_data)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            final_test_auc = test_auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
              f'Test: {test_auc:.4f}')

    print(f'Final Test: {final_test_auc:.4f}')

    # 1. Embedding-based link inference attacks
    # 1.1 similarity-based attack
    attack_edges = test_data.edge_label_index.numpy().T
    attack_edge_labels = test_data.edge_label.numpy()
    model.eval()
    embs = model.get_emb(train_data.x, train_data.edge_index).numpy()
    metric_list, attack_auc_list = sim_attacks(embs, attack_edges, attack_edge_labels)

    # 1.2 machine-learning-based attack
    shadow_edges = val_data.edge_label_index.numpy()
    shadow_edge_labels = val_data.edge_label.numpy()
    attack_edges = test_data.edge_label_index.numpy()
    attack_edge_labels = test_data.edge_label.numpy()
    model.eval()
    embs = model.get_emb(train_data.x, train_data.edge_index).numpy()
    ml_attacks(embs, shadow_edges, shadow_edge_labels, attack_edges, attack_edge_labels)

    # print(attack_res)

    # 2. Node attribute inference
    model.eval()
    embs = model.get_emb(train_data.x, train_data.edge_index)
    evaluator = MulticlassEvaluator()
    acc = log_regression(embs, train_data, evaluator, split='rand:0.1', num_epochs=3000)['acc']
    print("Node attr inference Acc: {:.4f}".format(acc))
