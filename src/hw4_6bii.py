import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("data/GolfTrends13March2026.csv")
    t_raw = df["Time"].values
    y = df["golf"].values

    num_months = []
    for time_str in t_raw:
        date = datetime.datetime.strptime(time_str, "%Y-%m-%d")
        months_since_2004 = (date.year - 2004) * 12 + (date.month - 1)
        num_months.append(months_since_2004)

    t = np.array(num_months) + 1
    n = len(t)

    X = np.zeros((n, n + 2))
    X[:, 0] = 1
    X[:, 1] = t - 1
    for j in range(2, n):
        X[:, j] = np.maximum(0, t - j)
    X[:, n] = np.cos(2 * np.pi * t / 12)
    X[:, n + 1] = np.sin(2 * np.pi * t / 12)

    P = np.zeros((n + 2, n + 2))
    np.fill_diagonal(P[2:n, 2:n], 1)

    # Define a uniform grid for gamma and sigma to evaluate the posterior
    gamma_grid = np.logspace(-2, 2, 150)  # spans 0.01 to 100
    sigma_grid = np.logspace(-1, 2, 150)  # spans 0.1 to 100
    log_posterior = np.zeros((len(gamma_grid), len(sigma_grid)))

    XtX = X.T @ X
    XtY = X.T @ y

    for i, gamma in enumerate(gamma_grid):
        A_g = XtX + (1 / gamma**2) * P

        # 1. Numerically stable log determinant using eigenvalues
        eigvals = np.linalg.eigvalsh(A_g)
        logdet_A = np.sum(np.log(np.maximum(eigvals, 1e-12)))

        # 2. Numerically stable solve
        beta_hat_g = np.linalg.lstsq(A_g, XtY, rcond=None)[0]

        # 3. Stable calculation of the Ridge loss / SSR
        # (Mathematically equal to yTy - XtY.T @ A_inv_XtY, but prevents floating point cancellation)
        SSR_g = np.sum((y - X @ beta_hat_g) ** 2) + (1 / gamma**2) * np.sum(
            beta_hat_g[2:n] ** 2
        )

        for j, sigma in enumerate(sigma_grid):
            # Using the derived marginal likelihood
            log_post = (
                -(n - 4) * np.log(sigma)
                - (n - 2) * np.log(gamma)
                - 0.5 * logdet_A
                - SSR_g / (2 * sigma**2)
            )
            log_posterior[i, j] = log_post

    # Exponentiate safely by subtracting the max value
    log_posterior -= np.max(log_posterior)
    posterior_probs = np.exp(log_posterior)
    posterior_probs /= np.sum(posterior_probs)

    # Draw N = 1000 samples from the 2D discrete grid
    N_samples = 1000
    flat_probs = posterior_probs.flatten()
    sampled_indices = np.random.choice(len(flat_probs), size=N_samples, p=flat_probs)
    sampled_i, sampled_j = np.unravel_index(sampled_indices, posterior_probs.shape)

    sampled_gamma = gamma_grid[sampled_i]
    sampled_sigma = sigma_grid[sampled_j]

    # Calculate point estimates (Posterior Means)
    gamma_hat = np.mean(sampled_gamma)
    sigma_hat = np.mean(sampled_sigma)
    lambda_equiv = 1 / (gamma_hat**2)

    print(f"Point estimate for gamma (gamma_hat): {gamma_hat:.4f}")
    print(f"Point estimate for sigma (sigma_hat): {sigma_hat:.4f}")
    print(f"Equivalent lambda (1 / gamma_hat^2): {lambda_equiv:.4f}")

    # Assuming X, y, t, P, sampled_gamma, and sampled_sigma are in the current namespace
    N = 1000
    n = len(y)

    # Initialize an array to store the sampled trend functions
    mu_samples = np.zeros((N, n))

    # Precompute X^T X and X^T y for efficiency
    XtX = X.T @ X
    XtY = X.T @ y

    for k in range(N):
        gamma_k = sampled_gamma[k]
        sigma_k = sampled_sigma[k]

        # Calculate A_gamma = X^T X + (1 / gamma^2) * D
        A_gamma = XtX + (1 / gamma_k**2) * P

        # Invert A_gamma to get the posterior covariance and mean
        A_inv = np.linalg.inv(A_gamma)
        mean_beta = A_inv @ XtY
        cov_beta = (sigma_k**2) * A_inv

        # Draw beta from the conditional Multivariate Normal posterior
        beta_k = np.random.multivariate_normal(mean_beta, cov_beta)

        # Compute the trend mu_t excluding the sinusoidal components
        # X[:, :n] grabs the intercept and all spline bases, beta_k[:n] grabs their coefficients
        mu_samples[k, :] = X[:, :n] @ beta_k[:n]

    # Generate the plot
    plt.figure(figsize=(12, 6))

    # Plot the 1000 posterior samples with very low opacity to show density
    for k in range(N):
        plt.plot(t, mu_samples[k, :], color="blue", alpha=0.02)

    # Overlay the original data points
    plt.scatter(t, y, color="black", s=15, label="Observed Data", zorder=5)

    plt.title("1000 Posterior Samples of the Trend Function (\mu_t)")
    plt.xlabel("Time (Months since Jan 2004)")
    plt.ylabel("Golf Search Interest")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Average over the N=1000 posterior samples (axis 0) to get the Bayesian point estimate
    mu_hat_bayesian = np.mean(mu_samples, axis=0)

    # Generate the comparison plot
    plt.figure(figsize=(12, 6))

    # Plot the raw observed data
    plt.scatter(t, y, color="black", s=15, label="Observed Data", zorder=5)

    # # Plot the Ridge estimate from part (a)
    # plt.plot(
    #     t,
    #     mu_hat_ridge,
    #     color="red",
    #     linestyle="--",
    #     alpha=0.7,
    #     label="Ridge Estimate (Part a)",
    # )

    # Plot the new Bayesian mean estimate
    plt.plot(
        t, mu_hat_bayesian, color="blue", linewidth=2, label="Bayesian Estimate (Part b)"
    )

    plt.title(
        "Comparison of Trend Estimates: Cross-Validated Ridge vs. Bayesian Posterior Mean"
    )
    plt.xlabel("Time (Months)")
    plt.ylabel("Golf Search Interest")
    plt.legend()
    plt.tight_layout()
    plt.show()

    beta_n_samples = np.zeros(N)
    beta_n1_samples = np.zeros(N)

    for k in range(N):
        gamma_k = sampled_gamma[k]

        # Calculate A_gamma = X^T X + (1 / gamma^2) * D
        A_gamma = XtX + (1 / gamma_k**2) * P

        # The conditional mean of beta is A_inv @ X^T y
        mean_beta = np.linalg.inv(A_gamma) @ XtY

        # The cosine and sine coefficients are the last two elements of beta
        beta_n_samples[k] = mean_beta[-2]
        beta_n1_samples[k] = mean_beta[-1]

    # Obtain the Bayesian point estimates (the averages)
    beta_n_hat = np.mean(beta_n_samples)
    beta_n1_hat = np.mean(beta_n1_samples)

    # Calculate the seasonal component
    seasonality = beta_n_hat * np.cos(2 * np.pi * t / 12) + beta_n1_hat * np.sin(
        2 * np.pi * t / 12
    )

    # Add the seasonal component to the smooth trend estimate from part (iii)
    full_estimate = mu_hat_bayesian + seasonality

    # Generate the plot
    plt.figure(figsize=(12, 6))
    plt.scatter(t, y, color="black", s=15, label="Observed Data", zorder=5)
    plt.plot(
        t,
        full_estimate,
        color="purple",
        linewidth=2,
        label="Full Bayesian Estimate (Trend + Seasonality)",
    )

    plt.title("Bayesian Posterior Mean Estimate with Seasonal Components")
    plt.xlabel("Time (Months)")
    plt.ylabel("Golf Search Interest")
    plt.legend()
    plt.tight_layout()
    plt.show()
