import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class CompositeLoss(torch.nn.Module):
    def __init__(self, lambda_sem=0.8, gamma_bal=1.5):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.gamma_bal = gamma_bal

    def modularity_loss(self, s, edge_index, num_nodes):
        """
        Minimizes inter-cluster coupling (Relaxed Normalized Cut).
        Optimizes for high intra-cluster density.
        """
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        degrees = torch.sum(adj, dim=1)
        d = torch.diag(degrees)
        s_t = s.t()
        
        # Maximize intra-cluster density => Minimize negative normalized trace
        intra_assoc = torch.trace(torch.matmul(torch.matmul(s_t, adj), s))
        degree_assoc = torch.trace(torch.matmul(torch.matmul(s_t, d), s))
        
        if degree_assoc == 0: return torch.tensor(0.0, device=s.device)
        return -1 * (intra_assoc / degree_assoc)

    def semantic_loss(self, s, x):
        """Maximizes semantic cohesion within clusters by penalizing distance to centroids."""
        # Calculate cluster centroids: (K, F)
        numerator = torch.matmul(s.t(), x)
        denominator = torch.sum(s, dim=0).unsqueeze(1) + 1e-10
        centroids = numerator / denominator
        
        # Average cosine distance to centroid
        loss = 0
        for k in range(s.shape[1]):
            cluster_prob = s[:, k]
            centroid = centroids[k]
            # Cosine similarity
            sim = F.cosine_similarity(x, centroid.unsqueeze(0), dim=1)
            # Weighted loss
            loss += torch.sum(cluster_prob * (1 - sim))
            
        return loss / x.shape[0]

    def balance_loss(self, s):
        """Uses KL Divergence to ensure balanced cluster sizes and avoid trivial solutions."""
        p = torch.mean(s, dim=0) # Current cluster distribution
        u = torch.ones(s.shape[1]).to(s.device) / s.shape[1] # Target Uniform distribution
        return F.kl_div(torch.log(p + 1e-10), u, reduction='sum')

    def forward(self, s, x, edge_index, num_nodes):
        l_mod = self.modularity_loss(s, edge_index, num_nodes)
        l_sem = self.semantic_loss(s, x)
        l_bal = self.balance_loss(s)
        
        # Weighted sum of losses
        total = l_mod + (self.lambda_sem * l_sem) + (self.gamma_bal * l_bal)
        return total, l_mod, l_sem, l_bal