from src.data.dataset import SyntheticFEMDatasetGenerator, save_dataset
from src.training.evaluator import load_checkpoint, predict_graphs
from src.training.trainer import train_model
from src.cli import main
from src.data.contract import fit_training_domain
import json
import pytest
import torch


def test_checkpoint_end_to_end(config, tmp_path, monkeypatch):
    config["dataset"]["path"] = str(tmp_path / "dataset.pt")
    config["training"]["epochs"] = 1
    save_dataset(SyntheticFEMDatasetGenerator(config).generate(), config["dataset"]["path"])
    _, metrics, path = train_model("gcn", config, tmp_path / "gcn")
    model, checkpoint = load_checkpoint(path)
    graphs = SyntheticFEMDatasetGenerator(config).generate()["test"]
    prediction, _ = predict_graphs(model, graphs)
    assert len(prediction) == 2
    assert checkpoint["epoch"] == 1 and "test" in metrics
    assert checkpoint["training_domain"] == fit_training_domain(SyntheticFEMDatasetGenerator(config).generate()["train"])
    assert checkpoint["checkpoint_version"] == 1
    history = json.loads((path.parent / "history.json").read_text())
    assert "stiffness" in history[0]["train_components"]
    assert history[0]["gradient_norm_max_before_clip"] > 0
    load_checkpoint(path, require_contract=True)
    report = tmp_path / "report"
    report.mkdir()
    summary = report / "metrics.json"
    summary.write_text('{"models": "existing experiment summary"}')
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--checkpoint", str(path),
                                   "--output", str(report)])
    main("evaluate")
    assert summary.read_text() == '{"models": "existing experiment summary"}'
    assert (report / "evaluation_metrics.json").exists()
    monkeypatch.setattr("sys.argv", ["cae", "candidates", "--config", "configs/smoke.yaml",
                                   "--count", "4", "--output", str(tmp_path / "candidates.json")])
    main()
    monkeypatch.setattr("sys.argv", ["cae", "screen", "--checkpoint", str(path),
                                   "--input", str(tmp_path / "candidates.json"), "--top-k", "2",
                                   "--output", str(tmp_path / "screen")])
    main()
    summary = json.loads((tmp_path / "screen/summary.json").read_text())
    assert summary["candidate_count"] == 4 and summary["checkpoint_sha256"]
    assert summary["fallback_count"] == 4  # GCN has no displacement head.
    checkpoint["feature_schema_sha256"] = "bad"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path)
