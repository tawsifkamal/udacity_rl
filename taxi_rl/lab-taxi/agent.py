import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, epsilon, alpha, gamma, env, nA=6):
    
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon
        self.env = env
        self.alpha = alpha
        self.gamma = gamma 

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Policy
        ====
        - The implemented policy is an epsilon greedy strategy with a decay rate of {} 

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if random.random() > self.epsilon: 
            action = np.argmax(self.Q[state])
        
        else:
            action = self.env.action_space.sample()
        
        return action

        

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done: 
            self.Q[state][action] += self.alpha * (reward + (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action])
        
        if done: 
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])


