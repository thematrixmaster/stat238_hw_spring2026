import numpy as np

data = np.loadtxt('data/PearsonHeightData.txt', skiprows=1)
X, Y = data[:, 0], data[:, 1]
n = len(X)
rho_hat = np.corrcoef(X, Y)[0, 1]
print(f'n = {n},  rho_hat = {rho_hat:.4f}')

rng = np.random.default_rng(42)
M = 10000

rho_boot = []
for _ in range(M):
    idx = rng.integers(0, n, size=n)
    rho_boot.append(np.corrcoef(X[idx], Y[idx])[0, 1])
rho_boot = np.array(rho_boot)
ci_boot = np.percentile(rho_boot, [2.5, 97.5])

rho_bayes = []
for _ in range(M):
    w = rng.exponential(1, size=n)
    w /= w.sum()
    mu_x = np.dot(w, X)
    mu_y = np.dot(w, Y)
    num = np.dot(w, (X - mu_x) * (Y - mu_y))
    den = np.sqrt(np.dot(w, (X - mu_x)**2) * np.dot(w, (Y - mu_y)**2))
    rho_bayes.append(num / den)

rho_bayes = np.array(rho_bayes)
ci_bayes = np.percentile(rho_bayes, [2.5, 97.5])

print(f"Classical Bootstrap 95% CI: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")
print(f"Bayesian Bootstrap 95% CI: [{ci_bayes[0]:.4f}, {ci_bayes[1]:.4f}]")
