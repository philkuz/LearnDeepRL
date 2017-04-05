from memory import Memory
from brain import Brain

import random
import numpy as np

MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

class Agent:
    def __init__(self, num_states, num_actions, eps_min=0.05, eps_max=1, lam=1e-3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.lam = lam
        self.brain = Brain(num_states, num_actions)
        self.memory = Memory(MEMORY_CAPACITY)
        self.step = 0

    def act(self, s):
        if random.random() < self.eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.brain.predict_one(s))

    def observe(self, sars_):
        '''
        takes in a sample of the environment, (s, a, r, s_),
        and adds it to the memory replay
        '''
        self.step += 1
        self.memory.add(sars_)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)

        states = batch[0]
        actions = batch[1]
        rewards = batch[2]
        states_ = batch[3]

        p = self.brain.predict(states)
        p_ = self.brain.predict(np.nan_to_num(states_))

        t = np.copy(p)
        t[:, actions] = rewards
        real_state = ~np.isnan(states_).any(axis=1)
        t[real_state,actions] += GAMMA * np.amax(p_, axis=1)[real_state]

        self.brain.train(states, t)

    @property
    def eps(self):
        return self.eps_min + (self.eps_max + self.eps_min) * np.exp(- self.lam  * self.step)

