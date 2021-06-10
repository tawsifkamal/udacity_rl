<<<<<<< HEAD
import numpy as np

print('hello world')

=======
import gym
import torch 
from collections import namedtuple
import random

from dqn_agent import Agent
from environment import Environment


# setting up our environment
env = Environment('LunarLander-v2')
agent = Agent(env.statespace, env.actionspace, 10)
# Setting up our agent 



# # # Defining the typle 
# # Experience = namedtuple('Experience', field_names= ['state', 'action', 'reward', 'next_state'])
# # new_tuple = Experience(state, action, reward, next_state)

# # print(new_tuple.action)

# state = env.reset()


# for i in range(200):
#     action = agent.act(state)
#     next_state, reward, done, info = env.step(action)
#     agent.step(state, action, reward, next_state, done)
    
#     if done == True:
#         break

# batch_experience = agent.memory.sample()


t1 = torch.randn((4, 3)).reshape(3, 4)
printn(t1)
>>>>>>> e9f129f3e3397843f146e48bf2f29e8d93e2ec82
