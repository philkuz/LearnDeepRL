from environment import Environment
from agent import Agent

PROBLEM = 'CartPole-v0'
env = Environment(PROBLEM)

num_states  = env.env.observation_space.shape[0]
num_actions = env.env.action_space.n

agent = Agent(num_states, num_actions)
try:
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("cartpole-basic.h5")

