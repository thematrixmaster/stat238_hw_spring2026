import numpy as np
from scipy import stats

# None indicates observations outside the measurement range
X = np.array([26.6, 38.5, 34.4, 34, 31, 23.6, None, None])
MIN, MAX = 23, 39

def calculate_densities(mu_grid: np.ndarray, log_sigma_grid: np.ndarray):
    # calculates posterior densities over grid points
    densities = np.zeros((len(mu_grid), len(log_sigma_grid)))
    for i, mu in enumerate(mu_grid):
        for j, log_sigma in enumerate(log_sigma_grid):
            sigma = np.exp(log_sigma)
            assert sigma > 0, "Sigma must be positive."
            Z_max = (MAX - mu) / sigma
            Z_min = (MIN - mu) / sigma
            posterior_density = 1.0 / sigma
            for x in X:
                if x is None:
                    prob_in_range = stats.norm.cdf(Z_max) - stats.norm.cdf(Z_min)
                    posterior_density *= (1 - prob_in_range)
                else:
                    Z = (x - mu) / sigma
                    posterior_density *= 1 / sigma * stats.norm.pdf(Z)
            densities[i, j] = posterior_density
    return densities

def normalize_densities(densities: np.ndarray):
    # numerically normalizes the 2D densities so that they sum to 1
    total_density = np.sum(densities)
    assert total_density > 0, "Total density must be positive"
    return densities / total_density

def mu_ci(normalized_densities: np.ndarray, mu_grid: np.ndarray, alpha=0.05):
    # calculate a 1-alpha credible interval for mu
    marginal_mu_density = np.sum(normalized_densities, axis=1)
    marginal_mu_density /= np.sum(marginal_mu_density)
    cumulative_density = np.cumsum(marginal_mu_density)
    lower_bound = mu_grid[np.searchsorted(cumulative_density, alpha / 2)]
    upper_bound = mu_grid[np.searchsorted(cumulative_density, 1 - alpha / 2)]
    print(f"95% c.i. for mu: [{lower_bound:.2f}, {upper_bound:.2f}]")

def main(mu_bounds, log_sigma_bounds):
    mu_grid = np.linspace(*mu_bounds, num=100)
    log_sigma_grid = np.linspace(*log_sigma_bounds, num=100)
    densities = calculate_densities(mu_grid, log_sigma_grid)
    normalized_densities = normalize_densities(densities)
    mu_ci(normalized_densities, mu_grid)

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    mu_bounds = (15, 50)
    log_sigma_bounds = (-1, 3)
    main(mu_bounds, log_sigma_bounds)
