import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DeepModuleNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters, heads=8):
        super(DeepModuleNet, self).__init__()
        
        # Structural Encoder
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

        # Clustering Head
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_clusters)
        )

    def forward(self, x, edge_index):
        # GAT Layers
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        embeddings = self.gat2(x, edge_index) # (N, out_channels)

        # Clustering
        logits = self.mlp(embeddings)
        s = F.softmax(logits, dim=1) # (N, K)
        
        return s, embeddings
