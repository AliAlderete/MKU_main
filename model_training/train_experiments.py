import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from starter.baseline_model import evaluate_regression_model


EXPERIMENTS = (
    {"fit_intercept": True, "positive": False},
    {"fit_intercept": False, "positive": False},
    {"fit_intercept": True, "positive": True},
)


def run_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    experiments: Sequence[Mapping[str, bool]] = EXPERIMENTS,
) -> list[dict[str, object]]:
    results = []

    for experiment_number, parameters in enumerate(experiments, start=1):
        model = LinearRegression(**parameters)
        model.fit(X_train, y_train)

        metrics = evaluate_regression_model(
            model,
            X_test,
            y_test,
            name=f"Experiment {experiment_number}",
        )
        result = {
            "experiment": experiment_number,
            "parameters": dict(parameters),
            "intercept": float(model.intercept_),
            "coefficients": model.coef_.tolist(),
            **metrics,
        }
        results.append(result)

        print(f"\n--- Experiment {experiment_number} ---")
        print(f"parameters: {json.dumps(result['parameters'], sort_keys=True)}")
        print(f"intercept: {result['intercept']:.6f}")
        print(f"coefficients: {json.dumps(result['coefficients'])}")
        print(f"test_mean_squared_error: {metrics['test_mean_squared_error']}")
        print(f"test_r_squared: {metrics['test_r_squared']}")

    return results


def main() -> None:
    np.random.seed(42)

    tips = sns.load_dataset("tips")
    df = pd.get_dummies(tips, drop_first=True)
    X = df.drop(columns=["tip"])
    y = df["tip"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metadata = {
        "train_shape": X_train_scaled.shape,
        "test_shape": X_test_scaled.shape,
        "num_features": X_train_scaled.shape[1],
    }
    print(f"Pipeline Data Diagnostics: {metadata}")

    X_train_sm = sm.add_constant(X_train_scaled)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(model_sm.summary())

    run_experiments(X_train_scaled, X_test_scaled, y_train, y_test)


if __name__ == "__main__":
    main()
