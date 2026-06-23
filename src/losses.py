"""Composite DeepModule objective from the revised manuscript."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CompositeLoss(torch.nn.Module):
    def __init__(self, lambda_sem: float = 0.7, gamma_bal: float = 0.1, beta_entropy: float = 0.05, eps: float = 1e-10):
        super().__init__()
        self.lambda_sem = float(lambda_sem)
        self.gamma_bal = float(gamma_bal)
        self.beta_entropy = float(beta_entropy)
        self.eps = float(eps)

    @staticmethod
    def dense_adjacency(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        if edge_index.numel() > 0:
            src = edge_index[0].long().to(device)
            dst = edge_index[1].long().to(device)
            adjacency[src, dst] = 1.0
        return adjacency

    def modularity_loss(self, s: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Negative directed modularity for soft assignments.

        Q_dir = (1/m) Tr(S^T B S), B_ij = A_ij - k_out_i k_in_j / m.
        """
        adjacency = self.dense_adjacency(edge_index, num_nodes, s.device)
        m = adjacency.sum()
        if m <= self.eps:
            return torch.zeros((), dtype=s.dtype, device=s.device)
        kout = adjacency.sum(dim=1, keepdim=True)
        kin = adjacency.sum(dim=0, keepdim=True)
        modularity_matrix = adjacency - (kout @ kin) / (m + self.eps)
        q_dir = torch.trace(s.t() @ modularity_matrix @ s) / (m + self.eps)
        return -q_dir

    def semantic_loss(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Semantic consistency loss (Eq. 12).

        L_sem = sum_i sum_k S_ik * (1 - cos(x_i, c_k)), averaged over the N
        nodes so the term is scale-stable across projects of different sizes.
        The per-node mean is a fixed positive rescaling of the manuscript sum
        and does not change the optimum; lambda is reported for this reduction.
        """
        centroids = (s.t() @ x) / (s.sum(dim=0).unsqueeze(1) + self.eps)
        x_norm = F.normalize(x, p=2, dim=1, eps=self.eps)
        c_norm = F.normalize(centroids, p=2, dim=1, eps=self.eps)
        cosine = x_norm @ c_norm.t()
        return torch.sum(s * (1.0 - cosine)) / x.shape[0]

    def balance_loss(self, s: torch.Tensor) -> torch.Tensor:
        p = s.mean(dim=0).clamp_min(self.eps)
        p = p / p.sum()
        u = torch.full_like(p, 1.0 / p.numel())
        kl = torch.sum(p * (torch.log(p) - torch.log(u)))
        entropy = -torch.sum(p * torch.log(p))
        return kl - self.beta_entropy * entropy

    def forward(self, s: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
        l_mod = self.modularity_loss(s, edge_index, num_nodes)
        l_sem = self.semantic_loss(s, x)
        l_bal = self.balance_loss(s)
        total = l_mod + self.lambda_sem * l_sem + self.gamma_bal * l_bal
        return total, l_mod, l_sem, l_bal
