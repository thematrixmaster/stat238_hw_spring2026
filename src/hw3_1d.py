import numpy as np
from scipy.stats import chi2

data = np.loadtxt('data/old_faithful_eruptions.txt')
n = len(data)
B = 1000

# Classical Bootstrap Interval
samples_c = np.random.choice(data, size=(B, n), replace=True)
sigma_class = np.std(samples_c, ddof=0, axis=1)
int_class = np.percentile(sigma_class, [2.5, 97.5])

# Bayesian Bootstrap Interval
weights = np.random.dirichlet(np.ones(n), size=B)
means_bayes = np.sum(weights * data, axis=1, keepdims=True)
sigma_bayes = np.sqrt(np.sum(weights * (data - means_bayes)**2, axis=1))
int_bayes = np.percentile(sigma_bayes, [2.5, 97.5])

# Normal Model Interval
ssd = np.sum((data - np.mean(data))**2)
lower_bound_norm = np.sqrt(ssd / chi2.ppf(0.975, df=n-1))
upper_bound_norm = np.sqrt(ssd / chi2.ppf(0.025, df=n-1))
int_normal = [lower_bound_norm, upper_bound_norm]

print(f"a) Classical Bootstrap 95% Interval: [{int_class[0]:.4f}, {int_class[1]:.4f}]")
print(f"b) Bayesian Bootstrap 95% Interval:  [{int_bayes[0]:.4f}, {int_bayes[1]:.4f}]")
print(f"c) Normal Model 95% Interval:        [{int_normal[0]:.4f}, {int_normal[1]:.4f}]")