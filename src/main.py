import argparse
import torch
import os
from src.data_loader import JavaProjectDataset
from src.trainer import DeepModuleTrainer

def main():
    parser = argparse.ArgumentParser(description="DeepModule: AI-driven Software Refactoring")
    parser.add_argument("--project_dir", type=str, required=True, help="Path to the Java project root directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--clusters", type=int, default=10, help="Number of target modules (clusters)")
    parser.add_argument("--output", type=str, default="refactoring_suggestions.csv", help="Output file for results")
    
    args = parser.parse_args()

    # 1. Data Preparation
    print(f"--- Processing Project: {args.project_dir} ---")
    dataset = JavaProjectDataset(root_dir=args.project_dir)
    dataset.parse_files()
    
    graph_path = "project_graph.pt"
    if os.path.exists(graph_path):
        print(f"Loading existing graph from {graph_path}...")
        data = torch.load(graph_path)
    else:
        data = dataset.build_graph()
        torch.save(data, graph_path)
        print(f"Graph saved to {graph_path}")

    # 2. Training
    print("--- Initializing Training ---")
    trainer = DeepModuleTrainer(data, num_clusters=args.clusters, device_name='cuda' if torch.cuda.is_available() else 'cpu')
    trainer.train(epochs=args.epochs)

    # 3. Inference & Saving
    print("--- Generating Recommendations ---")
    recommendations = trainer.predict()
    
    with open(args.output, "w") as f:
        f.write("Class,Predicted_Module\n")
        for cls, mod in recommendations.items():
            f.write(f"{cls},{mod}\n")
            
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
