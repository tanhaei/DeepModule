"""DeepModule neural architecture.

Uses torch_geometric GATConv when available. A pure-PyTorch attention fallback is
included so tests run in environments without torch-geometric.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv  # type: ignore
except Exception:  # pragma: no cover - expected in lightweight CI
    GATConv = None


class DenseGATLayer(nn.Module):
    """Small dense multi-head GAT layer for reproducible lightweight runs."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, concat: bool = True, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.weight = nn.Parameter(torch.empty(heads, in_channels, out_channels))
        self.att_src = nn.Parameter(torch.empty(heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(heads, out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(-1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        device = x.device
        # Add self-loops and build dense mask.
        loops = torch.arange(num_nodes, device=device)
        loop_edges = torch.stack([loops, loops], dim=0)
        all_edges = torch.cat([edge_index.to(device), loop_edges], dim=1)
        src, dst = all_edges[0], all_edges[1]
        mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        mask[src, dst] = True

        h = torch.einsum("ni,hio->hno", x, self.weight)  # heads, nodes, out
        src_score = torch.einsum("hno,ho->hn", h, self.att_src)
        dst_score = torch.einsum("hno,ho->hn", h, self.att_dst)
        e = F.leaky_relu(src_score.unsqueeze(2) + dst_score.unsqueeze(1), negative_slope=0.2)
        e = e.masked_fill(~mask.unsqueeze(0), float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.einsum("hij,hjo->hio", alpha, h)
        if self.concat:
            out = out.permute(1, 0, 2).reshape(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=0)
        return out, alpha.detach().mean(dim=0)


class DeepModuleNet(nn.Module):
    """GAT encoder + MLP soft clustering head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_clusters: int,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.uses_pyg = GATConv is not None
        self.dropout = dropout
        if self.uses_pyg:
            self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        else:
            self.gat1 = DenseGATLayer(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
            self.gat2 = DenseGATLayer(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_clusters),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.uses_pyg:
            x = F.elu(self.gat1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings = self.gat2(x, edge_index)
            self.last_attention = None
        else:
            x, _ = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            embeddings, attention = self.gat2(x, edge_index)
            self.last_attention = attention
        logits = self.mlp(embeddings)
        assignments = F.softmax(logits, dim=1)
        return assignments, embeddings
