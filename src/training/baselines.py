import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import regression_metrics, ranking_metrics
from .trainer import train_model
from ..data.dataset import load_dataset
from ..utils.config import resolve_path
from ..utils.logging import write_json


def train_baselines(config, output_dir="outputs/baselines"):
    dataset = load_dataset(resolve_path(config["dataset"]["path"]))
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=config.get("seed", 42), n_jobs=2
        ),
    }
    x = np.concatenate([g.design_parameters.numpy() for g in dataset["train"]])
    y = np.array([g.y_k.item() for g in dataset["train"]])
    results = {}
    for name, model in models.items():
        model.fit(x, np.log(y))
        results[name] = {}
        for split in ("val", "test", "ood"):
            features = np.concatenate([g.design_parameters.numpy() for g in dataset[split]])
            truth = np.array([g.y_k.item() for g in dataset[split]])
            prediction = np.exp(model.predict(features))
            results[name][split] = {**regression_metrics(truth, prediction),
                                     **ranking_metrics(truth, prediction)}
        joblib.dump(model, output_dir / f"{name}.joblib")
    _, metrics, _ = train_model("mlp", config, output_dir / "mlp")
    results["mlp"] = metrics
    write_json(output_dir / "metrics.json", {"provenance": "synthetic-only", "models": results})
    return results
