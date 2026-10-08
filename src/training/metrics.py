import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=float).ravel()
    prediction = np.asarray(prediction, dtype=float).ravel()
    if truth.shape != prediction.shape or not np.all(np.isfinite(prediction)):
        raise ValueError("Predictions must be finite and match target shape")
    return {
        "MAE": float(mean_absolute_error(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAPE_percent": float(100 * np.mean(np.abs(truth - prediction) / np.maximum(np.abs(truth), 1e-12))),
        "R2": float(r2_score(truth, prediction)) if len(truth) > 1 else None,
    }


def ranking_metrics(truth, prediction, top_k=(5, 10, 20)):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 1 or len(truth) < 1:
        raise ValueError("Ranking requires equal nonempty 1D arrays")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Ranking inputs must be finite")
    true_order = np.argsort(-truth, kind="stable")
    pred_order = np.argsort(-prediction, kind="stable")
    result = {}
    for k in top_k:
        if k < 1 or k > len(truth):
            continue
        result[f"Top{k}_Recall"] = float(len(set(true_order[:k]) & set(pred_order[:k])) / k)
    correlation = spearmanr(truth, prediction).statistic
    result["Spearman"] = float(correlation) if np.isfinite(correlation) else None
    return result
