from __future__ import annotations

import argparse
import csv
import os
import sys

from src.data_loader import JavaProjectDataset
from src.trainer import DeepModuleTrainer, estimate_module_count


def write_recommendations(path: str, recommendations: dict[str, int]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Class", "Recommended_Module"])
        for name, module in recommendations.items():
            writer.writerow([name, module])


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepModule: semantic-aware architecture modularization")
    parser.add_argument("--project_dir", type=str, required=True, help="Path to a Java project")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--clusters", type=int, default=5, help="Number of target modules K")
    parser.add_argument("--auto_k", action="store_true", help="Estimate K with the validation-free manuscript sweep")
    parser.add_argument("--ground_truth", type=str, default=None, help="CSV file with Class,True_Module columns")
    parser.add_argument("--min_loc", type=int, default=10, help="Minimum non-comment LOC for architecture entities")
    parser.add_argument("--no_codebert", action="store_true", help="Disable CodeBERT and use deterministic hashed embeddings")
    parser.add_argument("--language", choices=["java", "mixed"], default="java", help="Source language mode; mixed enables Java/Python/C++ regex parsing")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for output artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.clusters < 2 and not args.auto_k:
        print("Error: --clusters must be >= 2 unless --auto_k is used", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.project_dir):
        print(f"Error: directory not found: {args.project_dir}", file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print("=== Initializing DeepModule ===")
        dataset = JavaProjectDataset(
            root_dir=args.project_dir,
            min_non_comment_loc=args.min_loc,
            use_codebert=not args.no_codebert,
            language=args.language,
        )
        dataset.parse_files()
        data = dataset.build_graph()
        print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} directed edges")
        log_path = os.path.join(args.output_dir, "preprocessing_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file", "action", "reason"])
            writer.writeheader()
            writer.writerows(data.preprocessing_log)
        print(f"Preprocessing log saved to {log_path}")
    except Exception as exc:
        print(f"Fatal error in data preparation: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        selected_k = estimate_module_count(data, seed=args.seed) if args.auto_k else args.clusters
        if args.auto_k:
            print(f"Auto-selected K={selected_k}")
        trainer = DeepModuleTrainer(data, num_clusters=selected_k, seed=args.seed, output_dir=args.output_dir)
        trainer.train(epochs=args.epochs)
        recommendations = trainer.predict(write_behavior_artifacts=True)
        rec_path = os.path.join(args.output_dir, "modularization_recommendations.csv")
        write_recommendations(rec_path, recommendations)
        emb_path = trainer.save_embeddings()
        print(f"Recommendations saved to {rec_path}")
        print(f"Embeddings saved to {emb_path}")

        if args.ground_truth:
            if not os.path.exists(args.ground_truth):
                print(f"Warning: ground-truth file not found: {args.ground_truth}")
            else:
                ground_truth = JavaProjectDataset.load_ground_truth(args.ground_truth)
                metrics = trainer.evaluate(ground_truth)
                print("\n=== Evaluation Metrics (vs Expert Reference) ===")
                for key, value in metrics.items():
                    print(f"{key}: {value:.4f}")
    except Exception as exc:
        print(f"Fatal error during training/evaluation: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
