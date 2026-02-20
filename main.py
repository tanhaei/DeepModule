import argparse
import torch
import os
import sys
from src.data_loader import JavaProjectDataset
from src.trainer import DeepModuleTrainer

def main():
    parser = argparse.ArgumentParser(description="DeepModule: Modularization Recommendation Framework")
    parser.add_argument("--project_dir", type=str, required=True, help="Path to Java project")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--clusters", type=int, default=5, help="Number of target modules")
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to ground_truth.csv (Class,True_Module)")
    args = parser.parse_args()

    # Validation
    if args.clusters < 2:
        print("Error: clusters must be >= 2")
        sys.exit(1)
    if not os.path.exists(args.project_dir):
        print(f"Error: Directory '{args.project_dir}' not found.")
        sys.exit(1)

    try:
        print("=== Initializing DeepModule ===")
        dataset = JavaProjectDataset(root_dir=args.project_dir)
        dataset.parse_files()
        data = dataset.build_graph()
        print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    except Exception as e:
        print(f"Fatal Error in data preparation: {e}")
        sys.exit(1)

    trainer = DeepModuleTrainer(data, num_clusters=args.clusters)
    trainer.train(epochs=args.epochs)

    recommendations = trainer.predict()
    output_file = "modularization_recommendations.csv"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write("Class,Recommended_Module\n")
        for cls, mod in recommendations.items():
            f.write(f"{cls},{mod}\n")
    print(f"Recommendations saved to {output_file}")

    # Evaluation
    if args.ground_truth and os.path.exists(args.ground_truth):
        ground_truth = dataset.load_ground_truth(args.ground_truth)
        metrics = trainer.evaluate(ground_truth)
        print("\n=== Evaluation Metrics (vs Expert Ground Truth) ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()