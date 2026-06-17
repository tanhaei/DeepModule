"""Training, prediction, and evaluation for DeepModule."""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
torch.set_num_threads(1)
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from src.behavior_preservation import BehaviorPreserver
from src.losses import CompositeLoss
from src.model import DeepModuleNet


def estimate_module_count(data, max_k: Optional[int] = None, eta: float = 0.1, rho: float = 0.1, seed: int = 42) -> int:
    """Validation-free K selection from the manuscript.

    Sweeps K in {2, ..., ceil(sqrt(N))} and maximizes
    J(K)=Q_dir(K)+eta*semantic_silhouette(K)-rho*imbalance(K).
    A lightweight KMeans over semantic embeddings is used for the sweep.
    """
    from math import ceil, sqrt
    from sklearn.cluster import KMeans

    n = int(data.num_nodes)
    if n <= 2:
        return 2
    upper = max_k or int(ceil(sqrt(n)))
    upper = max(2, min(upper, n))
    x = data.x.detach().cpu().numpy()
    best_k = 2
    best_score = float("-inf")
    for k in range(2, upper + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)
        mq = DeepModuleTrainer.modularization_quality(data.edge_index, labels, n)
        sil = DeepModuleTrainer.semantic_silhouette(x, labels)
        counts = np.bincount(labels, minlength=k) / max(n, 1)
        imbalance = float(np.max(np.abs(counts - (1.0 / k))))
        score = mq + eta * sil - rho * imbalance
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k)


class DeepModuleTrainer:
    def __init__(
        self,
        data,
        num_clusters: int = 10,
        device_name: Optional[str] = None,
        seed: int = 42,
        lambda_sem: float = 0.7,
        gamma_bal: float = 0.1,
        beta_entropy: float = 0.05,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        heads: int = 8,
        learning_rate: float = 0.005,
        dropout: float = 0.0,
        output_dir: str = "outputs",
    ) -> None:
        self.seed = int(seed)
        self._set_seed(self.seed)
        self.device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.data = data.to(self.device)
        self.num_clusters = int(num_clusters)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if self.num_clusters < 2:
            raise ValueError("num_clusters must be >= 2")
        if self.data.num_nodes < self.num_clusters:
            raise ValueError(f"Classes ({self.data.num_nodes}) < clusters ({self.num_clusters})")

        self.model = DeepModuleNet(
            in_channels=self.data.x.shape[1],
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_clusters=self.num_clusters,
            heads=heads,
            dropout=dropout,
        ).to(self.device)
        self.criterion = CompositeLoss(lambda_sem=lambda_sem, gamma_bal=gamma_bal, beta_entropy=beta_entropy)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.history: List[dict] = []
        self.last_assignments: Optional[torch.Tensor] = None
        self.last_embeddings: Optional[torch.Tensor] = None

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def modularization_quality(edge_index: torch.Tensor, labels: np.ndarray, num_nodes: int) -> float:
        if edge_index.numel() == 0:
            return 0.0
        same = 0
        total = int(edge_index.shape[1])
        for src, dst in edge_index.t().cpu().numpy():
            if int(src) < num_nodes and int(dst) < num_nodes and labels[int(src)] == labels[int(dst)]:
                same += 1
        return same / max(total, 1)

    @staticmethod
    def mojo_fm_like(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
        """A reproducible 0--100 partition-agreement score.

        This is an executable proxy for MoJoFM based on optimal one-to-one module
        overlap. Full MoJoFM can be substituted if an external MoJo implementation
        is required for publication-scale experiments.
        """
        true = np.asarray(list(y_true), dtype=int)
        pred = np.asarray(list(y_pred), dtype=int)
        if true.size == 0:
            return 0.0
        true_labels = sorted(set(true.tolist()))
        pred_labels = sorted(set(pred.tolist()))
        matrix = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
        true_index = {label: i for i, label in enumerate(true_labels)}
        pred_index = {label: j for j, label in enumerate(pred_labels)}
        for t, p in zip(true, pred):
            matrix[true_index[int(t)], pred_index[int(p)]] += 1
        row_ind, col_ind = linear_sum_assignment(-matrix)
        matched = matrix[row_ind, col_ind].sum()
        return float(100.0 * matched / true.size)

    @staticmethod
    def semantic_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
        if len(set(labels.tolist())) < 2 or len(labels) <= len(set(labels.tolist())):
            return 0.0
        try:
            return float(silhouette_score(embeddings, labels, metric="cosine"))
        except Exception:
            return 0.0

    def train(self, epochs: int = 100, checkpoint_path: Optional[str] = None) -> List[dict]:
        checkpoint_path = checkpoint_path or os.path.join(self.output_dir, "model_checkpoint.pt")
        self.model.train()
        best_loss = float("inf")
        for epoch in range(int(epochs)):
            self.optimizer.zero_grad()
            assignments, embeddings = self.model(self.data.x, self.data.edge_index)
            loss, l_mod, l_sem, l_bal = self.criterion(assignments, self.data.x, self.data.edge_index, self.data.num_nodes)
            if torch.isnan(loss) or torch.isinf(loss):
                print("Invalid loss detected; rolling back to the last checkpoint.")
                if os.path.exists(checkpoint_path):
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            record = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "L_mod": float(l_mod.item()),
                "L_sem": float(l_sem.item()),
                "L_bal": float(l_bal.item()),
            }
            self.history.append(record)
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(self.model.state_dict(), checkpoint_path)
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                    f"Lmod: {l_mod.item():.4f} | Lsem: {l_sem.item():.4f} | Lbal: {l_bal.item():.4f}"
                )
        self._refresh_outputs()
        return self.history

    def _refresh_outputs(self) -> None:
        self.model.eval()
        with torch.no_grad():
            assignments, embeddings = self.model(self.data.x, self.data.edge_index)
        self.last_assignments = assignments.detach().cpu()
        self.last_embeddings = embeddings.detach().cpu()

    def predict(self, write_behavior_artifacts: bool = True) -> Dict[str, int]:
        if self.last_assignments is None:
            self._refresh_outputs()
        assert self.last_assignments is not None
        predictions = torch.argmax(self.last_assignments, dim=1).numpy()
        results = {name: int(predictions[i]) for i, name in enumerate(self.data.class_names)}
        if write_behavior_artifacts:
            preserver = BehaviorPreserver(self.data, results, output_dir=self.output_dir, attention_matrix=self.model.last_attention)
            preserver.write_boundary_report()
            preserver.generate_test_skeletons()
            preserver.lightweight_interface_check()
        return results

    def save_embeddings(self, filename: str = "embeddings.npy") -> str:
        if self.last_embeddings is None:
            self._refresh_outputs()
        assert self.last_embeddings is not None
        path = os.path.join(self.output_dir, filename)
        np.save(path, self.last_embeddings.numpy())
        return path

    def evaluate(self, ground_truth_dict: Dict[str, int]) -> Dict[str, float]:
        pred_dict = self.predict(write_behavior_artifacts=False)
        y_true: List[int] = []
        y_pred: List[int] = []
        used_names: List[str] = []
        for name, true_mod in ground_truth_dict.items():
            if name in pred_dict:
                used_names.append(name)
                y_true.append(int(true_mod))
                y_pred.append(int(pred_dict[name]))
        if len(y_true) < 2:
            return {"overlap_count": float(len(y_true))}
        labels = np.asarray([pred_dict[name] for name in self.data.class_names], dtype=int)
        embeddings = self.last_embeddings.numpy() if self.last_embeddings is not None else self.data.x.detach().cpu().numpy()
        return {
            "overlap_count": float(len(y_true)),
            "MoJoFM_proxy": self.mojo_fm_like(y_true, y_pred),
            "Adjusted_Rand_Index": float(adjusted_rand_score(y_true, y_pred)),
            "Normalized_Mutual_Info": float(normalized_mutual_info_score(y_true, y_pred)),
            "MQ": float(self.modularization_quality(self.data.edge_index, labels, self.data.num_nodes)),
            "Semantic_Silhouette": self.semantic_silhouette(embeddings, labels),
        }
