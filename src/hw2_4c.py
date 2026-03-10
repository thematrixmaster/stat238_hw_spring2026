import numpy as np
from scipy.special import logit

# 1. Define constants
n = 1295
x = 0
mu = -9.27308950
sigma2 = 0.09259668
sigma = np.sqrt(sigma2)

# 2. Set up the grid
# The range is small because expit(-9.27) is roughly 9.4e-5
theta = np.linspace(1e-9, 0.001, 100000)

# 3. Calculate Unnormalized Log Posterior
logit_theta = logit(theta)
log_prior = -np.log(theta * (1 - theta)) - 0.5 * ((logit_theta - mu) / sigma)**2
log_likelihood = x * np.log(theta) + (n - x) * np.log(1 - theta)
unnorm_log_post = log_likelihood + log_prior

# 4. Exponentiate and Normalize (with numerical stability shift)
unnorm_log_post -= np.max(unnorm_log_post) # Prevents underflow/overflow 
weights = np.exp(unnorm_log_post)
weights /= np.sum(weights)

# 5. Calculate Summary Statistics
post_mean = np.sum(theta * weights)
post_var = np.sum(((theta - post_mean)**2) * weights)

cumulative_weights = np.cumsum(weights)
lower_idx = np.searchsorted(cumulative_weights, 0.025)
upper_idx = np.searchsorted(cumulative_weights, 0.975)
cred_interval = (theta[lower_idx], theta[upper_idx])

# Output results
print(f"Posterior Mean: {post_mean:.8e}")
print(f"Posterior Variance: {post_var:.8e}")
print(f"95% Credible Interval: [{cred_interval[0]:.8e}, {cred_interval[1]:.8e}]")