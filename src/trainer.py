import torch
import torch.optim as optim
from src.model import DeepModuleNet
from src.losses import CompositeLoss

class DeepModuleTrainer:
    def __init__(self, data, num_clusters=10, device_name='cpu'):
        self.device = torch.device(device_name)
        self.data = data.to(self.device)
        self.num_clusters = num_clusters
        
        self.model = DeepModuleNet(
            in_channels=768, 
            hidden_channels=256, 
            out_channels=128, 
            num_clusters=num_clusters
        ).to(self.device)
        
        self.criterion = CompositeLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)

    def train(self, epochs=100, checkpoint_path="model_checkpoint.pt"):
        self.model.train()
        best_loss = float('inf')

        for epoch in range(epochs):
            try:
                self.optimizer.zero_grad()
                s, _ = self.model(self.data.x, self.data.edge_index)
                loss, l_mod, l_sem, l_bal = self.criterion(
                    s, self.data.x, self.data.edge_index, self.data.num_nodes
                )
                
                if torch.isnan(loss):
                    print("Error: NaN loss detected. Rolling back to previous state.")
                    break

                loss.backward()
                self.optimizer.step()

                if loss < best_loss:
                    best_loss = loss
                    torch.save(self.model.state_dict(), checkpoint_path)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"Training interrupted at epoch {epoch}: {e}")
                break

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            s, _ = self.model(self.data.x, self.data.edge_index)
            predictions = torch.argmax(s, dim=1)
        return {name: predictions[i].item() for i, name in enumerate(self.data.class_names)}