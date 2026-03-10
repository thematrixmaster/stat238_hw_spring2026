import numpy as np
import pandas as pd
from scipy.special import logsumexp
from hw2_7bc import log_marginal_poisson_gamma

def full_bayes_grid_posterior_theta_means(x, n, alpha_grid, beta_grid):
    log_post = np.empty((len(alpha_grid), len(beta_grid)))
    for i, a in enumerate(alpha_grid):
        for j, b in enumerate(beta_grid):
            log_post[i, j] = log_marginal_poisson_gamma(x, n, a, b) - np.log(a) - np.log(b)

    w = np.exp(log_post - logsumexp(log_post))    
    A = alpha_grid[:, None, None]
    B = beta_grid[None, :, None]    
    theta_mean_grid = (A + x) / (B + n)
    theta_post_mean = np.sum(w[..., None] * theta_mean_grid, axis=(0, 1))
    return theta_post_mean, (alpha_grid, beta_grid, w)

if __name__ == "__main__":
    d = pd.read_csv('KidneyCancerClean.csv')
    d['dct'] = d['dc'] + d['dc.2']
    d['popm'] = (d['pop'] + d['pop.2']) / 2
    x, n = d['dct'].values, d['popm'].values
    a_grid = np.geomspace(1, 100, 400)
    b_grid = np.geomspace(50000, 300000, 400)
    theta_post_mean, (alpha_grid, beta_grid, w) = full_bayes_grid_posterior_theta_means(x, n, a_grid, b_grid)
    print(theta_post_mean)