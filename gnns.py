import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim, cached=True,)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, cached=True)
        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def compute_loss(self, y_pred, y):
        loss = F.nll_loss(y_pred, y)
        return loss

    @torch.no_grad()
    def get_emb(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x

    @torch.no_grad()
    def test_model(self, logits, y, mask):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
        return acc


class UnsupGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UnsupGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def compute_loss(self, data):
        criterion = torch.nn.BCEWithLogitsLoss()
        z = self.encode(data.x, data.edge_index)

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            data.edge_label,
            data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = self.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        return loss

    @torch.no_grad()
    def get_emb(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    @torch.no_grad()
    def test_auc(self, data):
        z = self.encode(data.x, data.edge_index)
        out = self.decode(z, data.edge_label_index).view(-1).sigmoid()
        auc = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
        return auc


class UnsupSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(UnsupSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def compute_loss(self, batch):
        h = self.encode(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        return loss