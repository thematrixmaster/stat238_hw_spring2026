import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = np.array([26.6, 38.5, 34.4, 34, 31, 23.6, 120])

def main(mu_bounds):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,  # Set to True if you have TeX installed locally
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    n = len(X)
    mu_grid = np.linspace(*mu_bounds, num=1000)
    m_theta = np.abs(X[:, None] - mu_grid[None, :]).sum(axis=0)
    raw_posterior = (1.0 / m_theta) ** n

    dx = mu_grid[1] - mu_grid[0]
    normalized_densities = raw_posterior / (np.sum(raw_posterior) * dx)
    cumulative_density = np.cumsum(normalized_densities) * dx

    alpha = 0.05
    lower_bound = mu_grid[np.searchsorted(cumulative_density, alpha / 2)]
    upper_bound = mu_grid[np.searchsorted(cumulative_density, 1 - alpha / 2)]
    print(f"95% c.i. for mu: [{lower_bound:.2f}, {upper_bound:.2f}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    sns.lineplot(x=mu_grid, y=normalized_densities, ax=ax1, color="#1f77b4", linewidth=1.5)
    ax1.set_title("Posterior PDF")
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel("Density")

    sns.lineplot(x=mu_grid, y=cumulative_density, ax=ax2, color="#ff7f0e", linewidth=1.5)
    ax2.set_title("Posterior CDF")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel("Cumulative Prob.")

    plt.tight_layout()
    plt.savefig("posterior_plots.pdf", bbox_inches="tight")


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    mu_bounds = (15, 50)
    main(mu_bounds)
