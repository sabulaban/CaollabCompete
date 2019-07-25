from noise import OUNoise
from model_mod import Actor, Critic
import torch
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 20        # learning timestep interval
LEARN_NUM = 10          # number of learning passes
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter

class DDPGAgent:
	def __init__(self, state_size, action_size, random_seed, device, tau, n_agents = 2):
		self.device = device
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(random_seed)

		self.actor_local = Actor(state_size, action_size, random_seed = random_seed).to(self.device)
		self.actor_target = Actor(state_size, action_size, random_seed = random_seed).to(self.device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

		self.critic_local = Critic(n_agents * (action_size + state_size), random_seed = random_seed).to(self.device)
		self.critic_target = Critic(n_agents * (action_size + state_size), random_seed = random_seed).to(self.device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

		self.noise = OUNoise(action_size, random_seed, theta=OU_THETA, sigma=OU_SIGMA)

		self.tau = tau

		self.update(self.actor_local, self.actor_target)
		self.update(self.critic_local, self.critic_target)

	def act(self, state, noise = 0.0, target = False):
		state = torch.from_numpy(state).float().to(self.device)
		if target:
			actor = self.actor_target
		else:
			actor = self.actor_local
		self.actor_local.eval()
		with torch.no_grad():
			action = self.actor_local(state).cpu().data.numpy()
		self.actor_local.train()
		action = action + (noise * self.noise.sample())

		return np.clip(action, -1, 1)

	def update(self, local_model, target_model, soft = False, tau = 1.):
		if soft:
			tau = self.tau
#		else:
#			tau = ta
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




