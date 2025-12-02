import torch
import torch.optim as optim
from src.model import DeepModuleNet
from src.losses import CompositeLoss

class DeepModuleTrainer:
    def __init__(self, data, num_clusters=10, device_name='cpu'):
        self.device = torch.device(device_name)
        self.data = data.to(self.device)
        self.num_clusters = num_clusters
        
        # Initialize Model
        self.model = DeepModuleNet(
            in_channels=768, 
            hidden_channels=256, 
            out_channels=128, 
            num_clusters=num_clusters
        ).to(self.device)
        
        # Initialize Loss & Optimizer
        self.criterion = CompositeLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)

    def train(self, epochs=100):
        self.model.train()
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            s, _ = self.model(self.data.x, self.data.edge_index)
            
            loss, l_mod, l_sem, l_bal = self.criterion(
                s, self.data.x, self.data.edge_index, self.data.num_nodes
            )
            
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} (Mod: {l_mod.item():.2f}, Sem: {l_sem.item():.2f}, Bal: {l_bal.item():.2f})")
        
        print("Training finished.")

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            s, _ = self.model(self.data.x, self.data.edge_index)
            predictions = torch.argmax(s, dim=1)
            
        results = {}
        for i, class_name in enumerate(self.data.class_names):
            results[class_name] = predictions[i].item()
        return results
