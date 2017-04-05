import numpy as np

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.obs = None# observations in the algorithm

    def add(self, sars_):
        if self.obs:
            self.obs = np.vstack((self.obs, sars_))
        else:
            self.obs = sars_
        if len(self.obs) > self.capacity:
            self.obs = self.obs[1:]

    def sample(self, n):
        n = min(len(self.obs), n)
        idx = np.random.randint(len(self.obs), size=n)

        return self.obs[idx]
