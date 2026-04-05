import numpy as np
import pandas as pd
from scipy.special import gammaln, digamma

def log_evidence(theta):
    a, b = theta
    term1 = N * (gammaln(a + b) - gammaln(a) - gammaln(b))
    term2 = np.sum(gammaln(X + a) + gammaln(n_pop - X + b) - gammaln(n_pop + a + b))
    return term1 + term2

def grad_log_evidence(theta):
    a, b = theta
    grad_a = N * (digamma(a + b) - digamma(a)) + np.sum(digamma(X + a) - digamma(n_pop + a + b))
    grad_b = N * (digamma(a + b) - digamma(b)) + np.sum(digamma(n_pop - X + b) - digamma(n_pop + a + b))
    return np.array([grad_a, grad_b])

d = pd.read_csv('data/KidneyCancerClean.csv')
d['dct'] = d['dc'] + d['dc.2']
d['popm'] = (d['pop'] + d['pop.2']) / 2

X = d['dct'].values.astype(int)
theta = np.array([1.0, 1.0])
n_pop = d['popm']
N = len(X)

for it in range(3000):
    log_theta = np.log(theta)
    le = log_evidence(theta)
    g = grad_log_evidence(theta) * theta
    step = 0.01
    for _ in range(10):
        trial = np.exp(log_theta + step * g).clip(1e-10)
        if log_evidence(trial) > le:
            break
        step *= 0.5
    theta = np.exp(log_theta + step * g).clip(1e-10)
    if it % 100 == 0 or it == 2999:
        print(f"  iter {it:4d}: log_ev = {le:.2f}, a = {theta[0]:.4f}, b = {theta[1]:.4f}")

a_hat, b_hat = theta
print(f"\nFinal ML estimates: a = {a_hat:.4f}, b = {b_hat:.4f}")

d['bayes_posterior'] = (X + a_hat) / (n_pop + a_hat + b_hat)
d['bayes_high'] = d['bayes_posterior'] >= d['bayes_posterior'].nlargest(100).iloc[-1]

print("\nTop 5 counties by posterior rate:")
print(d[['dc', 'pop', 'bayes_posterior']].sort_values('bayes_posterior', ascending=False).head())