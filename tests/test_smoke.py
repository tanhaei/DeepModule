"""Smoke tests for the lightweight DeepModule pipeline."""

from __future__ import annotations

import os
import shutil
import tempfile

from generate_dummy_data import create_example_project
from src.data_loader import JavaProjectDataset
from src.trainer import DeepModuleTrainer, estimate_module_count, directed_modularity


def test_pipeline_smoke() -> None:
    tmp = tempfile.mkdtemp(prefix="deepmodule_test_")
    old = os.getcwd()
    try:
        os.chdir(tmp)
        create_example_project("example_project", "ground_truth.csv")
        dataset = JavaProjectDataset("example_project", use_codebert=False)
        dataset.parse_files()
        data = dataset.build_graph()
        assert data.num_nodes == 5
        assert data.edge_index.shape[1] >= 1
        k = estimate_module_count(data, seed=42)
        assert 2 <= k <= data.num_nodes
        trainer = DeepModuleTrainer(data, num_clusters=3, output_dir="outputs", seed=42)
        trainer.train(epochs=2)
        recs = trainer.predict()
        assert set(recs.keys()) == set(data.class_names)
        gt = JavaProjectDataset.load_ground_truth("ground_truth.csv")
        trainer.save_embeddings()
        metrics = trainer.evaluate(gt)
        assert metrics["overlap_count"] == 5.0
        assert os.path.exists("outputs/boundary_report.csv")
        assert os.path.exists("outputs/embeddings.npy")
    finally:
        os.chdir(old)
        shutil.rmtree(tmp, ignore_errors=True)


def test_directed_modularity_bounds() -> None:
    """Directed modularity should be higher for a correct partition than a
    scrambled one on a graph with clear community structure."""
    import numpy as np
    import torch

    # Two directed triangles: nodes 0,1,2 and 3,4,5.
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    good = np.array([0, 0, 0, 1, 1, 1])
    bad = np.array([0, 1, 0, 1, 0, 1])
    q_good = directed_modularity(edge_index, good, 6)
    q_bad = directed_modularity(edge_index, bad, 6)
    assert q_good > q_bad
    assert -1.0 <= q_good <= 1.0


def test_paired_significance() -> None:
    """The helper should report significance when DeepModule consistently beats
    the baseline across runs."""
    dm = [84.0, 84.2, 83.8, 84.1, 84.3, 83.9, 84.0, 84.2, 84.1, 83.7]
    base = [72.8, 73.0, 72.5, 72.9, 73.1, 72.6, 72.7, 73.0, 72.8, 72.4]
    result = DeepModuleTrainer.paired_significance(dm, base)
    assert result["mean_diff"] > 10.0
    assert result["p_value"] < 0.01
