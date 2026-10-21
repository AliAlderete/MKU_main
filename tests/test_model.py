from sklearn.dummy import DummyRegressor

from starter.baseline_model import evaluate_regression_model


def test_evaluate_regression_model_returns_metrics():
	model = DummyRegressor(strategy="constant", constant=0)
	model.fit([[0], [1]], [0, 0])

	metrics = evaluate_regression_model(
		model, [[0], [1]], [1, 3], name="Dummy"
	)

	assert metrics == {
		"model_name": "Dummy",
		"test_mean_squared_error": 5.0,
		"test_r_squared": -4.0,
	}

