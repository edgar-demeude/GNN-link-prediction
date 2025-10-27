import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
    

class GCNLinkPredictor(torch.nn.Module):
    """
    GCN pour la prédiction de liens avec décodage par Produit Scalaire ('dot') uniquement.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=3, dropout=0.3): # Décodeur  = 'dot'
        super().__init__()
        
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Couches intermédiaires (si num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
    
    def encode(self, x, edge_index):
        """Encode les nœuds en embeddings"""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def decode(self, z, edge_index, sigmoid=True):
        """Décode les embeddings en utilisant le produit scalaire (dot)"""
        src, dst = edge_index
        
        logits = (z[src] * z[dst]).sum(dim=-1)
        
        return torch.sigmoid(logits) if sigmoid else logits
    
    def decode_all(self, z, sigmoid=True):
        """Décode toutes les paires possibles (pour la matrice d'adjacence Ahat)"""
        logits = z @ z.T
        
        return torch.sigmoid(logits) if sigmoid else logits