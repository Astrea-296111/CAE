import pytest

from src.training.metrics import ranking_metrics, regression_metrics


def test_ranking_and_regression():
    truth = list(range(30))
    same = ranking_metrics(truth, truth)
    assert same["Top10_Recall"] == 1
    opposite = ranking_metrics(truth, truth[::-1])
    assert opposite["Top5_Recall"] == 0
    assert opposite["Spearman"] == pytest.approx(-1)
    assert regression_metrics([1, 2, 3], [1, 2, 3])["RMSE"] == 0
    with pytest.raises(ValueError):
        ranking_metrics(truth, [1])


def test_tail_errors_and_degenerate_targets():
    result = regression_metrics([100] * 10, list(range(100, 110)))
    assert result["MAPE_percent"] == pytest.approx(4.5)
    assert result["P90_relative_error_percent"] == pytest.approx(8.1)
    assert result["Max_relative_error_percent"] == 9
    assert result["R2"] is None
    with pytest.raises(ValueError):
        regression_metrics([float("nan")], [1])
    with pytest.raises(ValueError):
        regression_metrics([], [])
