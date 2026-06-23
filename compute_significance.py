"""Recompute the RQ1 significance test from the released per-run MoJoFM data.

This reproduces the statistical claim in Section 4.4 of the manuscript: the
per-project MoJoFM improvement of DeepModule over the strongest non-DeepModule
baseline is assessed with a paired Wilcoxon signed-rank test over the 10
repeated runs, and is significant at p < 0.01 on all six projects.

Usage:
    python compute_significance.py \
        --per_run data/releases/v3.0/benchmark_outputs/per_run_mojofm.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List

from src.trainer import DeepModuleTrainer


def load_per_run(path: str):
    dm: Dict[str, List[float]] = defaultdict(list)
    base: Dict[str, List[float]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            project = row["Project"]
            dm[project].append(float(row["DeepModule_MoJoFM"]))
            base[project].append(float(row["Strongest_Baseline_MoJoFM"]))
    return dm, base


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute RQ1 paired significance test")
    parser.add_argument(
        "--per_run",
        type=str,
        default="data/releases/v3.0/benchmark_outputs/per_run_mojofm.csv",
        help="CSV with Project,Run,DeepModule_MoJoFM,Strongest_Baseline_MoJoFM",
    )
    args = parser.parse_args()

    dm, base = load_per_run(args.per_run)
    print(f"{'Project':<10} {'mean_diff':>10} {'statistic':>10} {'p_value':>10} {'p<0.01':>8}")
    all_sig = True
    for project in dm:
        result = DeepModuleTrainer.paired_significance(dm[project], base[project])
        sig = result["p_value"] < 0.01
        all_sig = all_sig and sig
        print(
            f"{project:<10} {result['mean_diff']:>10.3f} {result['statistic']:>10.3f} "
            f"{result['p_value']:>10.5f} {('yes' if sig else 'no'):>8}"
        )
    print(f"\nAll six projects significant at p<0.01: {all_sig}")


if __name__ == "__main__":
    main()
