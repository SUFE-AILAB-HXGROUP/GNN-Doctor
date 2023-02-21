import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim, cached=True,
                             normalize=True)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim,
                             cached=True, normalize=True)
        self.conv3 = GCNConv(in_channels=hidden_dim, out_channels=output_dim,
                             cached=True, normalize=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def get_emb(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x


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

    @torch.no_grad()
    def get_emb(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x