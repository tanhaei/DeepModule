import torch
import torch.optim as optim
import os
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.model import DeepModuleNet
from src.losses import CompositeLoss

class DeepModuleTrainer:
    def __init__(self, data, num_clusters=10, device_name='cpu'):
        self.device = torch.device(device_name)
        self.data = data.to(self.device)
        self.num_clusters = num_clusters
        if self.data.num_nodes < self.num_clusters:
            raise ValueError(f"Classes ({self.data.num_nodes}) < clusters ({self.num_clusters})")
        
        self.model = DeepModuleNet(768, 256, 128, num_clusters).to(self.device)
        self.criterion = CompositeLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def train(self, epochs=100, checkpoint_path="model_checkpoint.pt"):
        self.model.train()
        best_loss = float('inf')
        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()
                s, _ = self.model(self.data.x, self.data.edge_index)
                loss, l_mod, l_sem, l_bal = self.criterion(s, self.data.x, self.data.edge_index, self.data.num_nodes)
                if torch.isnan(loss):
                    print("NaN loss → rollback")
                    if os.path.exists(checkpoint_path):
                        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    break
                loss.backward()
                self.optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.model.state_dict(), checkpoint_path)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Training error at epoch {epoch}: {e}")
                break

	def predict(self):
        results = {name: int(predictions[i]) for i, name in enumerate(self.data.class_names)}
        
        # === NEW: Behavior Preservation (Reviewer #2) ===
        from src.behavior_preservation import BehaviorPreserver
        preserver = BehaviorPreserver(self.data, results)
        preserver.generate_test_skeletons()
        preserver.differential_symbolic_check()
        
        return results

    def evaluate(self, ground_truth_dict):
        pred_dict = self.predict()
        y_true, y_pred = [], []
        for name, true_mod in ground_truth_dict.items():
            if name in pred_dict:
                y_true.append(true_mod)
                y_pred.append(pred_dict[name])
        if len(y_true) < 2:
            return {"warning": "Not enough overlapping classes"}
        return {
            "Adjusted_Rand_Index": adjusted_rand_score(y_true, y_pred),
            "Normalized_Mutual_Info": normalized_mutual_info_score(y_true, y_pred),
        }
