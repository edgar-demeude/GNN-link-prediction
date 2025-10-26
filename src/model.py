import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# VGAE is imported in the main notebook

class Encoder(torch.nn.Module):
    """
    2-layer GCN Encoder for the Variational Graph Autoencoder (VGAE).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)