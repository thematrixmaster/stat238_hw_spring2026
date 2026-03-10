from typing import List
from argparse import ArgumentParser

import numpy as np
from scipy import stats
from tqdm import tqdm


def generate(theta: float, var: float, threshold: float) -> List[float]:
    """Draws iid samples from a normal distribution with mean theta and variance var until
    an observation less than threshold is obtained. Returns the list of observations.

    Args:
        theta (float): mean of the normal
        var (float): variance of the normal
        threshold (float): threshold for stopping

    Returns:
        List[float]: list of observations drawn from the normal distribution
    """
    samples = []
    while True:
        sample = np.random.normal(theta, np.sqrt(var))
        samples.append(sample)
        if sample < threshold:
            break
    return samples


def main(args):
    """
    Runs the simulation experiment with a stopping rule, then calculates the confidence interval for the mean
    """
    n_success = 0

    for _ in tqdm(range(args.num_iter)):
        samples = generate(args.theta, args.std**2, args.threshold)
        N = len(samples)
        if N < 2:
            continue

        # calculate the sample mean
        sample_mean = np.mean(samples)

        # get the alpha/2 and 1-alpha/2 quantiles of the student-t distribution with N-1 degrees of freedom
        t_lower = stats.t.ppf(args.alpha / 2, df=N - 1)
        t_upper = stats.t.ppf(1 - args.alpha / 2, df=N - 1)
        assert np.isclose(t_lower, -t_upper), "The quantiles should be symmetric around zero for the student-t distribution."

        # calculate the lower and upper bounds of the confidence interval
        ci_lower = sample_mean + args.std * t_lower / np.sqrt(N)
        ci_upper = sample_mean + args.std * t_upper / np.sqrt(N)

        # check if the true mean theta is within the confidence interval
        if ci_lower <= args.theta <= ci_upper:
            n_success += 1

    coverage_probability = n_success / args.num_iter
    print(f"Estimated coverage probability: {coverage_probability:.4f}")
    print(f"Expected coverage probability: {1 - args.alpha:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate samples from a normal distribution until a threshold is met.")
    parser.add_argument("--theta", type=float, default=35.5, help="Mean of the normal distribution.")
    parser.add_argument("--std", type=float, default=5.5, help="Standard deviation of the normal distribution.")
    parser.add_argument("--threshold", type=float, default=25, help="Threshold for stopping the sampling.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for confidence interval.")
    parser.add_argument("--num-iter", type=int, default=100000, help="Number of iterations for the simulation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    np.random.seed(args.seed)
    main(args)
    
    breakpoint()
