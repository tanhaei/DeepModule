"""Smoke tests for the lightweight DeepModule pipeline."""

from __future__ import annotations

import os
import shutil
import tempfile

from generate_dummy_data import create_example_project
from src.data_loader import JavaProjectDataset
from src.trainer import DeepModuleTrainer, estimate_module_count


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
