import pymc as pm
import numpy as np

# 1. Define constants
n = 1295
x = 0
mu = -9.27308950
sigma = np.sqrt(0.09259668)

# 2. Build the PyMC Model
with pm.Model() as model:
    # Prior on the logit scale
    z = pm.Normal("z", mu=mu, sigma=sigma)
    
    # Deterministic transform back to probability scale [0, 1]
    theta = pm.Deterministic("theta", pm.math.invlogit(z))
    
    # Binomial Likelihood
    X_obs = pm.Binomial("X_obs", n=n, p=theta, observed=x)
    
    # 3. Sample from the posterior
    # random_seed added for reproducibility on your homework
    idata = pm.sample(draws=2000, tune=2000, chains=4, target_accept=0.95, random_seed=42)

# 4. Extract and calculate statistics
theta_draws = idata.posterior["theta"].values.flatten()
pymc_mean = np.mean(theta_draws)
pymc_var = np.var(theta_draws)
pymc_ci = np.quantile(theta_draws, [0.025, 0.975])

# Output results
print(f"PyMC Posterior Mean: {pymc_mean:.8e}")
print(f"PyMC Posterior Variance: {pymc_var:.8e}")
print(f"PyMC 95% Credible Interval: [{pymc_ci[0]:.8e}, {pymc_ci[1]:.8e}]")
