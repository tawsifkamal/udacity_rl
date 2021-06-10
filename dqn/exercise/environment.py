import gym 

class Environment():
    def __init__(self, env_name):
        #Defining our environment 
        self.env = gym.make(env_name)

        # Setting up our action and statespace 
        self.statespace = self.env.observation_space.shape[0]
        self.actionspace = self.env.action_space.n 

        print('Statespace: ', self.statespace)
        print('Actionspace: ', self.actionspace)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def step(self, action):
        return self.env.step(action)

    
    def run_environment(self, num_steps): 
        '''runs the environment for n time steps given an agent following a random policy. Also includes renderization of the environment 
        
        PARAMS
        ======

            num_steps(int): the number of timesteps that the user wants the agent to interact with the environment for 

         '''
        state = self.env.reset()

        for i in range(num_steps):
            reward_current_episode = 0
            track_t_steps = 0 
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.env.render()


            reward_current_episode += reward
            track_t_steps += i 
            state = next_state

            if done == True:
                break
        
        print('\nReward attained: ', reward_current_episode)
        print(f'Episode finished after {track_t_steps} timesteps. ')



