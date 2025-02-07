import numpy as np

class Bandit:
    def __init__(self, p):
        # p: Win rate
        self.p = p
        self.p_hat = 10
        # Samples collected so far
        self.N = 1

    def pull(self):
        return np.random.random() < self.p

    def update(self, x):
        self.N += 1
        self.p_hat = ((self.N - 1) * self.p_hat + x) / self.N

