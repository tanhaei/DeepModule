import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DeepModuleNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_clusters, heads=8):
        super(DeepModuleNet, self).__init__()
        
        # --- Layer 2: Structural Encoder (GAT) ---
        # First GAT layer with multi-head attention
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # Second GAT layer to aggregate neighborhood information
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

        # --- Layer 3: Soft Clustering Head ---
        # MLP projection to cluster space
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_clusters)
        )

    def forward(self, x, edge_index):
        # Forward pass through GAT layers
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        embeddings = self.gat2(x, edge_index) # (N, out_channels)

        # Project to cluster assignments
        logits = self.mlp(embeddings)
        s = F.softmax(logits, dim=1) # Soft assignment matrix (N, K)
        
        return s, embeddings