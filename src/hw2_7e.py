import pymc as pm
import numpy as np
import pandas as pd

def pymc_fit_loguniform_alpha_beta(x, n, draws=2000):
    N = len(x)
    with pm.Model() as model:
        log_alpha = pm.Flat("log_alpha")
        log_beta = pm.Flat("log_beta")        
        alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))
        beta = pm.Deterministic("beta", pm.math.exp(log_beta))        
        theta = pm.Gamma("theta", alpha=alpha, beta=beta, shape=N)        
        pm.Poisson("obs", mu=n * theta, observed=x)        
        idata = pm.sample(draws=draws, tune=2000, chains=4)
        
    theta_mean = idata.posterior["theta"].mean(dim=("chain", "draw")).values    
    return theta_mean, idata

if __name__ == "__main__":
    d = pd.read_csv('KidneyCancerClean.csv')
    d['dct'] = d['dc'] + d['dc.2']
    d['popm'] = (d['pop'] + d['pop.2']) / 2
    x, n = d['dct'].values, d['popm'].values
    theta_mean, idata = pymc_fit_loguniform_alpha_beta(x, n)
    print(theta_mean)