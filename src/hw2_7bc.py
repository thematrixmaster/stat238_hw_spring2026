import numpy as np
import pandas as pd
from scipy.special import gammaln
from IPython.display import display

def log_marginal_poisson_gamma(x, n, alpha, beta):
    ll = gammaln(alpha + x) - gammaln(alpha) - gammaln(x + 1)
    ll += x * np.log(n / (beta + n)) + alpha * np.log(beta / (beta + n))
    return np.sum(ll)

def fit_alpha_beta_grid_mle(x, n, a_grid, b_grid):
    LL = np.array([[log_marginal_poisson_gamma(x, n, a, b) for b in b_grid] for a in a_grid])
    idx = np.unravel_index(np.argmax(LL), LL.shape)
    return a_grid[idx[0]], b_grid[idx[1]], LL

def posterior_mean_theta(x, n, alpha, beta):
    return (alpha + x) / (beta + n)

if __name__ == "__main__":
    d = pd.read_csv('KidneyCancerClean.csv')
    d['dct'] = d['dc'] + d['dc.2']
    d['popm'] = (d['pop'] + d['pop.2']) / 2
    x, n = d['dct'].values, d['popm'].values
    a_grid = np.geomspace(1, 100, 400)
    b_grid = np.geomspace(50000, 300000, 400)
    alpha_hat, beta_hat, LL = fit_alpha_beta_grid_mle(x, n, a_grid, b_grid)
    print(f"alpha: {alpha_hat}, beta: {beta_hat}")
    a, b = 11.781889938777498, 120049.39668603126
    d['bayes1'] = (a + d['dct']) / (a + b + d['popm'])
    d['bayeshigh1'] = d['bayes1'] >= d['bayes1'].nlargest(100).iloc[-1]
    d['bayes2'] = posterior_mean_theta(x, n, alpha_hat, beta_hat)
    d['bayeshigh2'] = d['bayes2'] >= d['bayes2'].nlargest(100).iloc[-1]
    display(pd.crosstab(d['bayeshigh1'], d['bayeshigh2']))
    