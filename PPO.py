import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical

import numpy as np 



# 								PPO 
# This implementation works as follows: 
# Two networks, policy and values are created. They are separated networks. 
# Also, the clone policy network is created. It is used in the policy improvement to 
# prevent the updated policy to stray too far away from the old one.

# These two networks are then held in the PPO agent for ease of use. 
# The PPO agent keeps track of the visited states, actions taken, rewards and states values 
# At each time step, the think method is called to return an action from the policy.
# It also records the state, and the state value 

# After receiving the rewards, the observe method records the new state value, 
# the reward and a mask for terminal states


# Lastly, the train method is used to update all the networks according to the PPO algorithm 


class Policy(nn.Module): 
	
	def __init__(self, env_infos, hidden, lr): 

		nn.Module.__init__(self)
		self.hidden = hidden
		self.env_infos = env_infos 
		self.lr = lr 

		self.linears = nn.ModuleList()
		for i in range(len(hidden)): 
			if i == 0: 
				l = nn.Linear(env_infos[0], hidden[0])
			else:
				l = nn.Linear(hidden[i-1], hidden[i])
			self.linears.append(l)

		self.head = nn.Linear(hidden[-1], env_infos[1])

		self.adam = optim.Adam(self.parameters(), lr)

	
	def forward(self, x): 

		for l in self.linears: 
			x = F.relu(l(x))

		probs = F.softmax(self.head(x), dim = 1)
		return probs 

	def maj(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.parameters(),40.)
		self.adam.step()

	def release_clone(self):

		clone = Policy(self.env_infos, self.hidden, self.lr)
		clone.load_state_dict(self.state_dict())

		return clone 

	def eval_states(self, x, actions): 

		for l in self.linears: 
			x = F.relu(l(x))

		probs = F.log_softmax(self.head(x), dim = 1)

		selected = torch.gather(probs, dim = 1, index = actions)
		return selected


class Value(nn.Module): 
	
	def __init__(self, env_infos, hidden, lr): 

		nn.Module.__init__(self)
		self.hidden = hidden
		self.env_infos = env_infos 
		self.lr = lr 

		self.linears = nn.ModuleList()
		for i in range(len(hidden)): 
			if i == 0: 
				l = nn.Linear(env_infos[0], hidden[0])
			else:
				l = nn.Linear(hidden[i-1], hidden[i])
			self.linears.append(l)

		self.head = nn.Linear(hidden[-1], 1)

		self.adam = optim.Adam(self.parameters(), lr)

	
	def forward(self, x): 

		for l in self.linears: 
			x = F.relu(l(x))

		probs = self.head(x)

		return probs 

	def maj(self, loss): 

		self.adam.zero_grad()
		loss.backward()
		self.adam.step()

class PPO(): 

	def __init__(self, env_infos, policy_param, value_param, clone_update = 2): 

		self.policy = Policy(env_infos, policy_param[0], policy_param[1])
		self.value = Value(env_infos, value_param[0], value_param[1])

		self.clone = self.policy.release_clone()

		self.actions = []
		self.visited_states = []
		self.v_mem = []
		self.rewards = []

		self.is_training = True

		self.clone_update = clone_update
		self.counter = 0 

	def clear(self): 

		self.actions = []
		self.visited_states = []
		self.v_mem = []
		self.rewards = []

	def eval(self): 

		self.is_training = False 

	def demo(self, x): 

		probs = self.policy(x).data.numpy().reshape(-1)
		return np.argmax(probs)

	def demo_non_tensor(self,x): 

		x = torch.tensor(x).float().reshape(1,-1)
		return self.demo(x)

	def follow(self, x, action): 

		probs = self.policy(x)
		m = Categorical(probs)

		self.actions.append(torch.LongTensor([action]))
		self.visited_states.append(x)
		self.v_mem.append([self.value(x)])

	def think(self, x): 

		x.requires_grad_(True)
		probs = self.policy(x)
		m = Categorical(probs)
		action = m.sample()


		if(self.is_training): 

			self.actions.append(action)
			self.visited_states.append(x)
			self.v_mem.append([self.value(x)])

		return action.item()

	def observe(self, new_state, reward, done): 

		mask = 0. if done else 1. 
		next_val = self.value(new_state).data[0]*mask

		self.v_mem[-1].append(next_val)
		self.rewards.append(reward)

	def get_advantage(self): 

		avantages, retours = [],[]
		a = 0
		r = 0 

		for i in reversed(range(len(self.rewards))): 

			r = r*0.99 + self.rewards[i]
			d = r + 0.99*self.v_mem[i][1].item() - 0.99*self.v_mem[i][0].item()
			a = d + 0.99*0.95*a 

			avantages.insert(0,a)
			retours.insert(0,r)

		avantages = torch.tensor(avantages)
		avantages = (avantages - avantages.mean())/(avantages.std() + 1e-6)

		return avantages.view(-1,1), torch.tensor(retours).view(-1,1)


	def train(self): 

		avantages, retours = self.get_advantage()

		# train critic

		next_estims = np.zeros((len(self.v_mem),1))
		for i in range(next_estims.shape[0]): 
			next_estims[i,0] = self.v_mem[i][1]

		next_estims = torch.tensor(next_estims).float()
		# next_estims = torch.cat([self.v_mem[i][1] for i in range(len(self.v_mem))]).view(-1,1)
		target = retours + 0.99*next_estims
		# target.requires_grad_(True)

		estims = torch.cat([self.v_mem[i][0] for i in range(len(self.v_mem))]).view(-1,1)

		value_loss = F.mse_loss(estims, target)
		self.value.maj(value_loss)

		# train actor 
		states = torch.cat(self.visited_states)
		actions = torch.cat(self.actions).view(-1,1)

		if states.shape[0] > 1: 
			old_prob = self.clone.eval_states(states, actions)
			new_prob = self.policy.eval_states(states, actions)

			ratio = torch.exp(new_prob - old_prob)
			
			# avantages.requires_grad_(True)
			surr_1 = ratio*avantages
			surr_2 = torch.clamp(ratio, 0.8,1.2)*avantages
			policy_loss = -(torch.min(surr_1,surr_2)).mean()

			if self.counter%self.clone_update == 0:
				self.clone = self.policy.release_clone()
			self.policy.maj(policy_loss)
		
		self.clear()
		self.counter = (self.counter +1)%self.clone_update

# import gym 

# env = gym.make('CartPole-v0')
# params = [[64],5e-3]
# player = PPO([4,2],params,params)

# epochs = 1000 
# mean_r = 0 
# for epoch in range(epochs): 

# 	s = env.reset()
# 	done = False 
# 	reward = 0 
# 	while not done: 

# 		action = player.think(torch.tensor(s).reshape(1,-1).float())
# 		ns, r, done, _ = env.step(action)

# 		player.observe(torch.tensor(ns).float(), r, done)

# 		s = ns 
# 		reward += r

# 		if done: 	

# 			mean_r += reward
# 			player.train()

# 			if epoch%100 == 0: 
# 				print('It {}/{} -- Mean reward {:.3f} '.format(epoch, epochs, mean_r/100.))
# 				mean_r = 0.