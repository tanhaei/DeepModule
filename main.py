import argparse
import torch
import os
import sys
from src.data_loader import JavaProjectDataset
from src.trainer import DeepModuleTrainer

def main():
    parser = argparse.ArgumentParser(description="DeepModule: AI-driven Software Refactoring")
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--clusters", type=int, default=5)
    args = parser.parse_args()

    # Step 1: Data Preparation with Error Handling
    try:
        dataset = JavaProjectDataset(root_dir=args.project_dir)
        dataset.parse_files()
        data = dataset.build_graph()
    except Exception as e:
        print(f"Fatal Error during initialization: {e}")
        sys.exit(1)

    # Step 2: Training with state recovery
    print("--- Starting Optimization ---")
    trainer = DeepModuleTrainer(data, num_clusters=args.clusters)
    trainer.train(epochs=args.epochs)

    # Step 3: Inference
    try:
        recommendations = trainer.predict()
        output_file = "refactoring_suggestions.csv"
        with open(output_file, "w") as f:
            f.write("Class,Predicted_Module\n")
            for cls, mod in recommendations.items():
                f.write(f"{cls},{mod}\n")
        print(f"Success! Results saved to {output_file}")
    except Exception as e:
        print(f"Error during recommendation generation: {e}")

if __name__ == "__main__":
    main()