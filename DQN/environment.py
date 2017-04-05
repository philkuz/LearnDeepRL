import gym
import numpy as np

class Environment:

    def __init__(self, problem):
        self.env = gym.make(problem)


    def run(self, agent):
        s = self.env.reset()
        R=  0
        while True:
            a = agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:
                s_ = np.nan # replace None with nan so that it will be matched by numpy stuff

            agent.observe( np.array([[s, a, r, s_]]))
            agent.replay()

            s = s_
            R += r

            if done:
                break
        print("Total Reward: ", R)

