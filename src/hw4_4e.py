import pandas as pd
import numpy as np

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

    cdf = np.cumsum(post)
    ci_lower = c_grid[np.searchsorted(cdf, 0.025)]
    ci_upper = c_grid[np.searchsorted(cdf, 0.975)]

    print(f"95% CI for c: [{ci_lower:.2f}, {ci_upper:.2f}]")
