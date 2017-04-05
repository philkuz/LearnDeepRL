import numpy as np

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.s = None
        self.a = None
        self.r = None
        self.s_ = None

    def add(self, sars_):
        ''' Add sars_ to the memory '''
        s = np.reshape(sars_[0], (1, len(sars_[0])))
        a = np.array([sars_[1]])
        r = np.array([sars_[2]])
        s_ = np.reshape(sars_[3], (1, len(sars_[3])))
        if self.s is None:
            self.s = s
            self.a = a
            self.r = r
            self.s_ = s_
        else:
            self.s = np.vstack((self.s, s))
            self.a = np.vstack((self.a, a))
            self.r = np.vstack((self.r, r))
            self.s_ = np.vstack((self.s_, s_))

        # remove the observation if its greater than the capacity
        if len(self.s) > self.capacity:
            self.s = self.s[1:]
            self.a = self.a[1:]
            self.r = self.r[1:]
            self.s_ = self.s_[1:]

    def sample(self, n):
        n = min(len(self.s), n)
        idx = np.random.randint(len(self.s), size=n)
        s = self.s[idx]
        a = self.a[idx]
        r = self.r[idx]
        s_ = self.s_[idx]
        return (s, a, r, s_)
