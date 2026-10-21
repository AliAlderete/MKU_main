import joblib
import numpy as np
import pandas as pd

from model_training.train_experiments import run_experiments, train_best_model


def test_run_experiments_tracks_parameters_and_metrics(capsys):
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([9.0, 11.0])

    results = run_experiments(X_train, X_test, y_train, y_test)
    output = capsys.readouterr().out

    assert len(results) == 3
    assert results[0]["parameters"] == {
        "fit_intercept": True,
        "positive": False,
    }
    assert results[0]["test_mean_squared_error"] == 0.0
    assert "parameters:" in output
    assert "coefficients:" in output
    assert "test_r_squared:" in output


def test_train_best_model_saves_the_lowest_mse_model(tmp_path):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = np.array([3.0, 5.0, 7.0])
    results = [
        {
            "test_mean_squared_error": 2.0,
            "parameters": {"fit_intercept": True, "positive": False},
        },
        {
            "test_mean_squared_error": 1.0,
            "parameters": {"fit_intercept": False, "positive": False},
        },
    ]
    artifact_path = tmp_path / "model" / "best_model.joblib"

    train_best_model(X, y, results, artifact_path)

    artifact = joblib.load(artifact_path)
    assert artifact["parameters"] == {
        "fit_intercept": False,
        "positive": False,
    }
    assert artifact["feature_columns"] == ["feature"]
    assert artifact["model"].n_features_in_ == 1
