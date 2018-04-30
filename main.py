from PPO import PPO
from Tilter import World, Render 
import numpy as np 
import torch 
import arcade

# Creating environnement

env = World(continuous = False)


env_spaces = env.infos() # observation space, action space 

# param: hidden layers and learning rate 
policy_params = [[32,32],1e-3]
value_params = [[32,32],1e-3]
agent = PPO(env_spaces, policy_params, value_params, clone_update = 10) # defining the agent. The clone is the old_policy


# Learning loop 
epochs = 10000
mean_reward = 0
for epoch in range(epochs): 

	s = env.reset()
	done = False 
	reward = 0 

	while not done: 

		action = agent.think(torch.tensor(s).float().reshape(1,-1)) # returns the action 
		s, r, done = env.step(action)
		agent.observe(torch.tensor(s).float().reshape(1,-1),r, done) # records the transition 

		reward += r 
		if done: 
			mean_reward += reward
			agent.train()

			if epoch%100 == 0: 
				print('{}/{} -- Mean reward {:.3f} '.format(epoch, epochs, mean_reward/100.))
				mean_reward = 0.

# Visualize the simulation: the render uses the environnement and the agent
r = Render(env, agent)
arcade.run()