from unityagents.environment import UnityEnvironment
from dqn_agent import Agent
from unityagents import UnityEnvironment
import torch


env = UnityEnvironment(file_name='Banana_Windows_x86_64/Banana.exe', worker_id=234)
brain = env.brains[env.brain_names[0]]
env_info = env.reset(train_mode=False)[env.brain_names[0]]
state = env_info.vector_observations[0]
observation_space = len(state)
action_space = 4 


agent = Agent(observation_space, action_space, 10)

env.close()