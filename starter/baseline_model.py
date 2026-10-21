import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_regression_model(model, X, y_true, name="Model"):
    predictions = model.predict(X)
    mse = mean_squared_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    metrics = {
        "model_name": name,
        "test_mean_squared_error": round(mse, 4),
        "test_r_squared": round(r2, 4)
    }
    return metrics

def main():
    np.random.seed(42)

    tips = sns.load_dataset('tips')
    df = pd.get_dummies(tips, drop_first=True)
    X = df.drop(columns=['tip'])
    y = df['tip'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metadata = {
        "train_shape": X_train_scaled.shape,
        "test_shape": X_test_scaled.shape,
        "num_features": X_train_scaled.shape[1]
    }
    print(f"Pipeline Data Diagnostics: {metadata}")

    X_train_sm = sm.add_constant(X_train_scaled)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(model_sm.summary())

    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)

    baseline_metrics = evaluate_regression_model(
        reg, X_test_scaled, y_test, name="Scikit-Learn OLS"
    )
    print("--- Standard Evaluation Output ---")
    for metric, value in baseline_metrics.items():
        print(f"{metric}: {value}")


<<<<<<< HEAD
print("--- Testing the triggering of CI actions ---")
print("--- A ---")
print("--- Testing the triggering of CI actions ---")
=======
if __name__ == "__main__":
    main()
>>>>>>> 2a48ca1 (	new file:   starter/__pycache__/baseline_model.cpython-314.pyc)
