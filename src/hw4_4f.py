import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dt = pd.read_csv("data/wagedata.csv")
    y = np.log(dt["WeeklyEarnings"].values)
    x = dt["Exper"].values
    n, p = len(y), 3

    c_grid = np.linspace(2.1, 59.9, 1000)
    log_post = np.zeros(len(c_grid))

    for i, c in enumerate(c_grid):
        Xc = np.column_stack((np.ones(n), x, np.maximum(0, x - c)))
        XtX = Xc.T @ Xc
        beta = np.linalg.solve(XtX, Xc.T @ y)
        rss = np.sum((y - Xc @ beta) ** 2)
        log_post[i] = -0.5 * np.linalg.slogdet(XtX)[1] - ((n - p) / 2) * np.log(rss)

    post = np.exp(log_post - np.max(log_post))
    post /= np.sum(post)

    # --- Ancestral Sampling & Plotting ---
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.2, color="gray", s=10, label="Data")

    x_plot = np.linspace(x.min(), x.max(), 500)

    for j in range(100):
        # 1. Sample c ~ Categorical(post)
        c_samp = np.random.choice(c_grid, p=post)

        # Calculate dependencies for c_samp
        Xc = np.column_stack((np.ones(n), x, np.maximum(0, x - c_samp)))
        XtX_inv = np.linalg.inv(Xc.T @ Xc)
        beta_hat = XtX_inv @ Xc.T @ y
        rss = np.sum((y - Xc @ beta_hat) ** 2)

        # 2. Sample sigma^2 ~ Inv-Gamma(shape, rate) equivalent to 1 / Gamma(shape, scale=1/rate)
        sigma2_samp = 1 / np.random.gamma(shape=(n - p) / 2, scale=1 / (rss / 2))

        # 3. Sample beta ~ MVN
        beta_samp = np.random.multivariate_normal(beta_hat, sigma2_samp * XtX_inv)

        # Evaluate and plot curve
        y_plot = (
            beta_samp[0]
            + beta_samp[1] * x_plot
            + beta_samp[2] * np.maximum(0, x_plot - c_samp)
        )
        plt.plot(x_plot, y_plot, color="blue", alpha=0.1)
        plt.scatter(c_samp, beta_samp[0] + beta_samp[1] * c_samp, color="red", zorder=5)

    plt.xlabel("Years of Experience (x)")
    plt.ylabel("Log Weekly Earnings (y)")
    plt.title("Posterior Samples of Piecewise Linear Fit")
    plt.show()
