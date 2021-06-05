import torch
import torch.nn as nn
import torch.nn.functional as F

import gym 

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(in_features = state_size, out_features = 32)
        self.out = nn.Linear(32, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = F.relu(self.fc1(state))
        state = self.out(state)
        return state 

# Defining the Environment 
env = gym.make('LunarLander-v2')
statespace = env.observation_space.shape[0]
actionspace = env.action_space.n
model = QNetwork(statespace, actionspace)