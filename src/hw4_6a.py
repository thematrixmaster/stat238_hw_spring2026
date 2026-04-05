import pandas as pd
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

if __name__ == "__main__":
    df = pd.read_csv("data/GolfTrends13March2026.csv")
    t_raw = df["Time"].values
    y = df["golf"].values

    num_months = []
    for time_str in t_raw:
        date = datetime.datetime.strptime(time_str, "%Y-%m-%d")
        months_since_2004 = (date.year - 2004) * 12 + (date.month - 1)
        num_months.append(months_since_2004)

    # Align time to start at t=1 to match the equation t=1...n
    t = np.array(num_months) + 1
    n = len(t)

    # 1. Construct feature matrix X and penalty matrix P
    X = np.zeros((n, n + 2))
    X[:, 0] = 1  # beta_0
    X[:, 1] = t - 1  # beta_1
    for j in range(2, n):  # beta_2 to beta_{n-1}
        X[:, j] = np.maximum(0, t - j)
    X[:, n] = np.cos(2 * np.pi * t / 12)  # beta_n
    X[:, n + 1] = np.sin(2 * np.pi * t / 12)  # beta_{n+1}

    P = np.zeros((n + 2, n + 2))
    np.fill_diagonal(P[2:n, 2:n], 1)  # Penalize only beta_2 to beta_{n-1}

    # 2. Cross-validation for lambda
    lambdas = np.logspace(-1, 5, 50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_errors = []

    for lam in lambdas:
        fold_err = 0
        for train_idx, test_idx in kf.split(X):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            # Closed-form ridge solution: beta = (X^T X + lambda*P)^-1 X^T Y
            beta = np.linalg.solve(X_tr.T @ X_tr + lam * P, X_tr.T @ y_tr)
            fold_err += np.sum((y_te - X_te @ beta) ** 2)
        cv_errors.append(fold_err)

    best_lam = lambdas[np.argmin(cv_errors)]
    print(f"Optimal Regularization Parameter (lambda): {best_lam:.2f}")

    # 3. Fit final model with optimal lambda
    beta_hat = np.linalg.solve(X.T @ X + best_lam * P, X.T @ y)

    # 4. Compute trend functions
    mu_hat = X[:, :n] @ beta_hat[:n]  # Base trend without seasonality
    y_hat = X @ beta_hat  # Full model with seasonality

    # 5. Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(t, y, "o", color="lightgray", label="Original Data")
    plt.plot(t, mu_hat, "r-", linewidth=2, label=r"$\hat{\mu}_t$ (Trend only)")
    plt.plot(
        t, y_hat, "b-", linewidth=2, alpha=0.7, label=r"$\hat{\mu}_t +$ Seasonality"
    )
    plt.title("Golf Search Trends: P-Spline Ridge Regression")
    plt.xlabel("Months ($t$)")
    plt.ylabel("Search Interest")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
