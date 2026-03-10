import numpy as np
from scipy import stats

X = np.array([26.6, 38.5, 34.4, 34, 31, 23.6])
MIN, MAX = 23, 39

def calculate_densities(mu_grid: np.ndarray, log_sigma_grid: np.ndarray):
    """scans over the grid of mu and log_sigma values, calculate the joint posterior unnormalized density
    for each pair of (mu, log_sigma), and returns a 2D array of densities.
    """
    densities = np.zeros((len(mu_grid), len(log_sigma_grid)))
    for i, mu in enumerate(mu_grid):
        for j, log_sigma in enumerate(log_sigma_grid):
            sigma = np.exp(log_sigma)
            assert sigma > 0, "Sigma must be positive."
            Z = (X-mu)/sigma
            Z_max = (MAX-mu)/sigma
            Z_min = (MIN-mu)/sigma
            # calculate the density of the truncated normal distribution for each observation in X
            denom = stats.norm.cdf(Z_max) - stats.norm.cdf(Z_min)
            if denom == 0:
                continue
            trun_norm_densities = 1/sigma * stats.norm.pdf(Z) / denom
            posterior_density = 1/sigma * np.prod(trun_norm_densities)
            densities[i, j] = posterior_density
    return densities

def normalize_densities(densities: np.ndarray):
    """numerically normalizes the 2D array of densities so that they sum to 1, and returns the normalized densities."""
    total_density = np.sum(densities)
    if total_density == 0:
        raise ValueError("Total density is zero, cannot normalize.")
    return densities / total_density

def mu_ci(normalized_densities: np.ndarray, mu_grid: np.ndarray, alpha: float = 0.05):
    # calculate a 1-alpha credible interval for mu
    marginal_mu_density = np.sum(normalized_densities, axis=1)
    marginal_mu_density /= np.sum(marginal_mu_density)  # normalize the marginal density
    cumulative_density = np.cumsum(marginal_mu_density)
    lower_bound = mu_grid[np.searchsorted(cumulative_density, alpha / 2)]
    upper_bound = mu_grid[np.searchsorted(cumulative_density, 1 - alpha / 2)]
    print(f"95% credible interval for mu: [{lower_bound:.2f}, {upper_bound:.2f}]")

def main(mu_bounds, log_sigma_bounds):
    mu_grid = np.linspace(*mu_bounds, num=100)
    log_sigma_grid = np.linspace(*log_sigma_bounds, num=100)
    densities = calculate_densities(mu_grid, log_sigma_grid)
    normalized_densities = normalize_densities(densities)
    mu_ci(normalized_densities, mu_grid, alpha=0.05)

if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    mu_bounds = (15,50)
    log_sigma_bounds = (-1,3)
    main(mu_bounds, log_sigma_bounds)
