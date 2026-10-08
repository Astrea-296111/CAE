from src.data.dataset import SyntheticFEMDatasetGenerator, save_dataset
from src.training.evaluator import load_checkpoint, predict_graphs
from src.training.trainer import train_model
from src.cli import main


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
    report = tmp_path / "report"
    report.mkdir()
    summary = report / "metrics.json"
    summary.write_text('{"models": "existing experiment summary"}')
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--checkpoint", str(path),
                                   "--output", str(report)])
    main("evaluate")
    assert summary.read_text() == '{"models": "existing experiment summary"}'
    assert (report / "evaluation_metrics.json").exists()
