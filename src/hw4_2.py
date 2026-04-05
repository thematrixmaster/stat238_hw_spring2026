import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

def get_W(X, y, beta):
    xTb = X @ beta
    Psi = norm.cdf(xTb)
    psi = norm.pdf(xTb)
    return psi * (y * (psi + xTb * Psi) / Psi**2 + (1 - y) * (psi - xTb * (1 - Psi)) / (1 - Psi)**2)

def neg_log_like(beta, X, y):
    xTb = X @ beta
    ll = y * norm.logcdf(xTb) + (1 - y) * norm.logsf(xTb)
    return -1 * np.sum(ll)

def neg_gradient(beta, X, y):
    xTb = X @ beta
    phi = norm.pdf(xTb)
    Phi = norm.cdf(xTb)
    term1 = y * phi / np.maximum(Phi, 1e-10)
    term2 = (1 - y) * phi / np.maximum(1 - Phi, 1e-10)
    grad = (term1 - term2) @ X
    return -grad

if __name__ == "__main__":
    frogs = pd.read_csv("data/frogs.csv")
    y = frogs["pres.abs"].astype(int)

    X = pd.DataFrame(
        {
            "altitude": frogs["altitude"],
            "log_distance": np.log(frogs["distance"]),
            "log_NoOfPools": np.log(frogs["NoOfPools"]),
            "NoOfSites": frogs["NoOfSites"],
            "avrain": frogs["avrain"],
            "meanmin": frogs["meanmin"],
            "meanmax": frogs["meanmax"],
        }
    )
    X = sm.add_constant(X)

    # Frequentist approach
    probit_model = sm.Probit(y, X).fit()
    beta_hat = probit_model.params.values
    std_error = probit_model.bse.values

    # Bayesian approach (laplace posterior approximation)
    W = np.diag(get_W(X.values, y, beta_hat))

    Xm = X.values
    Hess = -(Xm.T @ (W @ Xm))
    cov_beta = np.linalg.inv(-Hess)
    posterior_std = np.sqrt(np.diag(cov_beta))

    # check if the standard errors are close within a reasonable tolerance
    assert np.allclose(std_error, posterior_std, atol=1e-2), \
        "Standard errors from frequentist and Bayesian approaches are not close enough. Something might be wrong."

    # Calculate \hat{beta} using Newton-Raphson method
    beta_newton = np.zeros(X.shape[1])
    for iteration in range(20):
        # Calculate Gradient and Hessian at current beta
        xTb = Xm @ beta_newton
        phi = norm.pdf(xTb)
        Phi = norm.cdf(xTb)

        grad = ((y / Phi - (1 - y) / (1 - Phi)) * phi) @ Xm # Gradient (Score vector)
        W = get_W(Xm, y, beta_newton)                       # Hessian
        H = -(Xm.T @ (np.diag(W) @ Xm))  

        # Newton Update: beta = beta - H^-1 * grad
        step = np.linalg.solve(H, grad)
        beta_newton -= step

        if np.linalg.norm(step) < 1e-6:
            print(f"Newton converged in {iteration} iterations")
            break

    assert np.allclose(beta_newton, beta_hat, atol=1e-2), \
        "Newton-Raphson beta estimates do not match Probit model estimates. Something might be wrong."
