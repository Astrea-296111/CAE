from src.data.dataset import SyntheticFEMDatasetGenerator, save_dataset
from src.training.evaluator import load_checkpoint, predict_graphs
from src.training.trainer import train_model


def test_checkpoint_end_to_end(config, tmp_path):
    config["dataset"]["path"] = str(tmp_path / "dataset.pt")
    config["training"]["epochs"] = 1
    save_dataset(SyntheticFEMDatasetGenerator(config).generate(), config["dataset"]["path"])
    _, metrics, path = train_model("gcn", config, tmp_path / "gcn")
    model, checkpoint = load_checkpoint(path)
    graphs = SyntheticFEMDatasetGenerator(config).generate()["test"]
    prediction, _ = predict_graphs(model, graphs)
    assert len(prediction) == 2
    assert checkpoint["epoch"] == 1 and "test" in metrics
