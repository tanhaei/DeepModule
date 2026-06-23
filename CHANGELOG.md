# Changelog — manuscript-alignment revision

This revision aligns the implementation with the revised manuscript and the
response-to-reviewers document.

## Fixed

1. **K-selection now uses directed modularity (Eq. 1 / Eqs. 8--9).**
   `estimate_module_count` previously scored candidate K with the intra-cluster
   edge fraction (MQ). It now uses the directed modularity `Q_dir` defined in the
   manuscript, computed by the new `directed_modularity()` helper. This was
   verified to match the loss-side `Tr(S^T B S)/m` matrix form exactly.

2. **Semantic consistency loss documented against Eq. 12.**
   `CompositeLoss.semantic_loss` computes `sum_i sum_k S_ik (1 - cos(x_i, c_k))`
   reduced by a per-node mean (`/N`). The mean is a fixed positive rescaling that
   does not change the optimum and stabilizes training across project sizes; this
   is now stated explicitly in the docstring.

3. **Statistical-significance claim (Section 4.4) is now reproducible.**
   Added `DeepModuleTrainer.paired_significance()` (paired Wilcoxon signed-rank),
   the `compute_significance.py` script, and the released per-run data
   (`per_run_mojofm.csv`) and results (`statistical_tests.csv`). Recomputation
   yields p = 0.00195 < 0.01 on all six projects, supporting the manuscript.

4. **Cross-language reference construction (Section 4.9) is now documented.**
   The BigCode Java/Python/C++ subsets have no expert references; their reference
   modularization is derived from the original top-level package/directory
   structure. This is recorded in the release `README.md`, a new `Reference_Source`
   column in `cross_language_results.csv`, and the `MANIFEST.json`.

## Tests

- `tests/test_smoke.py` extended with `test_directed_modularity_bounds` and
  `test_paired_significance`.
- Full pipeline (`generate_dummy_data.py` -> `main.py`), `--auto_k`,
  `--language mixed`, and `compute_significance.py` were executed end-to-end.
