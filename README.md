# DeepModule: Semantic-Aware Architecture Modularization

DeepModule is an unsupervised framework for architecture-level modularization of monolithic Java systems. It builds a directed software dependency graph, enriches nodes with semantic embeddings, learns dependency-aware representations with a Graph Attention Network, and outputs candidate module/microservice boundaries for human review.

## Scope

- DeepModule recommends architecture-level module boundaries.
- It does not automatically transform source code.
- It does not guarantee runtime migration properties such as latency, data ownership, distributed transactions, or eventual consistency.
- The boundary report and generated JUnit skeletons are review artifacts, not formal behavior-equivalence proofs.

## Manuscript-aligned implementation details

- Directed dependency graph from imports, type references, object creation, and inheritance.
- Semantic node vectors from CodeBERT when available, with deterministic hashed embeddings as an offline fallback.
- Two-layer GAT encoder with 8 attention heads in the first layer.
- Differentiable soft clustering head.
- Composite objective: directed modularity + semantic consistency + cluster balance.
- Defaults: `lambda=0.7`, `gamma=0.1`, `beta=0.05`, `epochs=100`, `lr=0.005`.
- Explicit preprocessing log for excluded/generated/test/vendor/short files.

## Install

```bash
pip install -r requirements.txt
```

The lightweight smoke test runs without `transformers`, `javalang`, or `torch-geometric`; those packages are optional for full experiments.

## Smoke test

```bash
python generate_dummy_data.py
python main.py --project_dir ./example_project --clusters 3 --epochs 5 --ground_truth ground_truth.csv --no_codebert --output_dir outputs_smoke
```

Expected outputs:

- `outputs_smoke/modularization_recommendations.csv`
- `outputs_smoke/embeddings.npy`
- `outputs_smoke/boundary_report.csv`
- `outputs_smoke/test_skeletons/`

## Cross-language smoke mode

For BigCode-style mixed subsets, use the lightweight regex parser mode:

```bash
python main.py --project_dir /path/to/subset --language mixed --auto_k --no_codebert
```

A full Tree-sitter/CodeBERT setup can be plugged in for publication-scale experiments; the offline fallback keeps CI deterministic.

## Full run

```bash
python main.py --project_dir /path/to/java/project --clusters 12 --epochs 100 --ground_truth expert_reference.csv --output_dir outputs
```

## Statistical significance (RQ1)

The per-project MoJoFM improvement over the strongest non-DeepModule baseline is
assessed with a paired Wilcoxon signed-rank test over the 10 repeated runs
(Section 4.4). Recompute it from the released per-run data with:

```bash
python compute_significance.py --per_run data/releases/v3.0/benchmark_outputs/per_run_mojofm.csv
```

## Manuscript v3.0 replication package

The manuscript-level replication package is available under:

```text
data/releases/v3.0
```

It contains source-code links, expert-rubric summaries, anonymized expert-feedback files, MoJoFM comparison outputs, component-ablation outputs, cross-language generalization outputs, boundary-analysis evidence, figure data, and output schemas.
