from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent(epsilon = 1, alpha = 0.04, gamma = 0.9, env = env)
avg_rewards, best_avg_reward = interact(env, agent)

