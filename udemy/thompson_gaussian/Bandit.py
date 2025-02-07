import numpy as np


class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean  # The true (unknown) mean of the bandit
        self.m = 0                  # Prior estimate of the mean (initial guess)
        self.lambda_ = 1            # Prior precision (inverse of variance)
        self.tau = 1                # Known precision of the observations (fixed)
        self.N = 0                  # Counter for the number of observations

    # Generate a random sample from the true underlying normal distribution N(true_mean, 1/tau)
    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    # Sample from the current estimated posterior distribution N(m, 1/lambda_)
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1

