# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import ujson 
import gym
import numpy as np
from collections import defaultdict

from plot_utils import plot_blackjack_values, plot_policy, plot_rewards
import matplotlib.pyplot as plt 


# %%
env = gym.make('Blackjack-v0')


# %%
print(env.observation_space)
print(env.action_space)


# %%
def generate_episode_pol(env, Q, epsilon):
    state = env.reset()
    episode = []
    while True:
        # in the first time-step we have to explore lol 
        action = eps_greedy_policy(epsilon, Q, state)
        next_state, reward, done, info = env.step(action)
        state = next_state 
        episode.append((state, action, reward))
        if done == True:
            break 
    
    return episode

def eps_greedy_policy(epsilon, Q, state):
    probs = np.zeros(2)
    optimal_action = np.argmax(Q[state])
    sub_optimal_action = abs(optimal_action - 1)
    probs[optimal_action] = 1 - epsilon + (epsilon / 2) 
    probs[sub_optimal_action] = epsilon / 2
    
    action = np.random.choice(np.arange(2), p = probs)
    
    return action 
    
def MC_control(env, num_episodes, gamma=1):
# initialize Q to zero for all states and actions
# initialize the visit counter for state-action pair to zero for all states and actions 
    Q = defaultdict(lambda: np.zeros(2))
    epsilon = 1
    rewards_all_episodes = []

    
# from 1 to num episodes, update epsilon via the episode count that you are currently in 
    for i in range(1, num_episodes + 1):
        epsilon  = max(0.05, epsilon * 0.99999) 
        if i % 1000 == 0:
            print("\rEpisode {}/{}.".format(i, num_episodes), end="")
            sys.stdout.flush()
# another for loop generating episodes following the current policy pi 
        episode = generate_episode_pol(env, Q, epsilon)
        states, actions, rewards = zip(*episode)
        rewards = np.array(rewards)
        rewards_all_episodes.append(sum(rewards))
        
        
# from again t = 1 to the terminal time step
        for i, state in enumerate(states):
            discounts = np.array([gamma ** i for i in range(len(rewards[i: ]))])
            current_return = np.sum(rewards[i: ] * discounts)
            Q[state][actions[i]] += (current_return - Q[state][actions[i]]) * 0.02
            policy = dict((state, np.argmax(q_value)) for state, q_value in Q.items())
    
    rewards_all_episodes = np.array(rewards_all_episodes)    
    return Q, policy, rewards_all_episodes          


# %%
num_episodes = 1000000
Q, policy, rewards = MC_control(env, num_episodes, gamma = 1.0)
rewards = np.array([0 if x == -1 else x for x in rewards])
rewards = rewards.cumsum()
win_rate = rewards / np.arange(1, num_episodes+1)
plt.scatter(np.arange(1, num_episodes+1), win_rate)


# %%
file = open('q_values.json')
Q = ujson.load(file)
file.close()
# %%
def generate_episode_optimal(env, Q):
    state = env.reset()
    episode = []
    while True:
        # in the first time-step we have to explore lol 
        action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        state = next_state 
        episode.append((state, action, reward))
        if done == True:
            break 
    
    return episode

def plotting()
    


# %%
rewards_all_episodes = []
num_episodes = 50000

for episode in range(1, num_episodes + 1):
    experience = generate_episode_optimal(env, Q)
    if episode % 1000 == 0:
        print("\rEpisode {}/{}.".format(episode, num_episodes), end="")
        sys.stdout.flush()
            
    state, action, reward_per_episode = zip(*experience)
    rewards_all_episodes.append(sum(reward_per_episode))
    plot_rewards()


# %%
rewards = np.array(rewards_all_episodes)
rewards = np.array([0 if x == -1 else x for x in rewards])
rewards = rewards.cumsum()
# win_rate = rewards / np.arange(1, num_episodes+1)
# plt.scatter(np.arange(1, num_episodes+1), win_rate)
# plt.show()


# %%
win_rate = rewards / np.arange(1, num_episodes+1)
plt.scatter(np.arange(1, num_episodes+1), win_rate)


# %%
# obtain the corresponding state-value function
V = dict((k,np.max(v)) for k, v in Q.items())

# plot the state-value function
plot_blackjack_values(V)


# %%
# plot the policy
plot_policy(policy)

# %% [markdown]
# ![True Optimal Policy](optimal.png)

# %%



