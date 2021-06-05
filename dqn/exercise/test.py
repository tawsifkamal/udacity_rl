import gym

env = gym.make('LunarLander-v2')
state = env.reset()

# for i in range(100):
#     reward_current_episode = 0 
#     action = env.action_space.sample()
#     next_state, reward, done, info = env.step(action)
#     reward_current_episode += reward 
#     env.render()

#     if done == True:
#         print(f'Rewards: {reward_current_episode}')
#         break

# env.close()

print(env.observation_space.shape)