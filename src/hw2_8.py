import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp

baseball = pd.DataFrame({
    "players": [
        "Clemente", "F Robinson", "F Howard", "Johnstone", "Berry",
        "Spencer", "Kessinger", "L Alvarado", "Santo", "Swoboda",
        "Unser", "Williams", "Scott", "Petrocelli", "E Rodriguez",
        "Campaneris", "Munson", "Alvis"
    ],
    "hits": [
        18, 17, 16, 15, 14, 14, 13, 12, 11,
        11, 10, 10, 10, 10, 10, 9, 8, 7
    ],
    "atbats": [45] * 18,
    "EoSaverage": [
        0.346, 0.298, 0.276, 0.222, 0.273, 0.270, 0.263, 0.210,
        0.269, 0.230, 0.264, 0.256, 0.303, 0.264, 0.226,
        0.286, 0.316, 0.200
    ]
})

# James–Stein estimator
n = 45
baseball["norm_data"] = 2 * np.sqrt(n) * np.arcsin(np.sqrt(baseball["hits"] / n))
baseball["true_mean"] = 2 * np.sqrt(n) * np.arcsin(np.sqrt(baseball["EoSaverage"]))

mu = baseball["norm_data"].mean()
fctr = 1 - (len(baseball["norm_data"]) - 3) / np.sum((baseball["norm_data"] - mu) ** 2)
baseball["js"] = mu + (baseball["norm_data"] - mu) * fctr
# James–Stein estimator

H, n = baseball["hits"].values, baseball["atbats"].iloc[0]
theta_true = baseball["EoSaverage"].values

loga, logb = np.linspace(-6, 6, 181), np.linspace(-6, 6, 181)
A, B = np.exp(loga)[:, None], np.exp(logb)[None, :]

logpost = sum(betaln(A + h, B + n - h) for h in H) - len(H) * betaln(A, B)
w = np.exp(logpost - logsumexp(logpost)).ravel()

rng = np.random.default_rng(238)
idx = rng.choice(w.size, size=30000, p=w)
a_s, b_s = np.exp(loga[idx // 181]), np.exp(logb[idx % 181])

theta_samps = rng.beta(a_s[:, None] + H, b_s[:, None] + n - H)

bayes_mean = theta_samps.mean(axis=0)
ci_lo, ci_hi = np.quantile(theta_samps, [0.025, 0.975], axis=0)

naive_p = H / n
js_p = np.sin(baseball["js"] / (2 * np.sqrt(n))) ** 2

SSE = lambda est: np.sum((est - theta_true)**2)
SAE = lambda est: np.sum(np.abs(est - theta_true))

metrics = pd.DataFrame({
    "Estimator": ["Naive", "James-Stein", "Bayes"],
    "SSE": [SSE(naive_p), SSE(js_p), SSE(bayes_mean)],
    "SAE": [SAE(naive_p), SAE(js_p), SAE(bayes_mean)]
})

coverage = np.sum((theta_true >= ci_lo) & (theta_true <= ci_hi))

print(metrics)
print(f"95% interval coverage count: {coverage} out of {len(H)}")
