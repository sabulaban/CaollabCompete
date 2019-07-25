from ddpg import DDPGAgent
import torch
from memory_mod import ReplayBuffer
import numpy as np
import torch.nn.functional as F
from collections import deque

BUFFERSIZE = int(1e6)
BATCHSIZE = 256

class MADDPG:
	def __init__(self, device, state_size, action_size, random_seed = 12, gamma=0.95, tau=0.02, n_agents = 2, numUpdates = 3, learnSteps = 1, cheater = False, epsilon = 1., epsilon_decay = 1e-6, debug = False):
		self.device = device
		self.gamma = gamma
		self.tau = tau

		self.state_size = state_size
		self.action_size = action_size
		self.n_agents = n_agents

		self.random_seed = random_seed

		self.numUpdates = numUpdates
		self.learnSteps = learnSteps

		#state_size, action_size, random_seed, device, tau, n_agents = 2

		self.agents = [DDPGAgent(self.state_size, self.action_size, self.random_seed, self.device, self.tau, self.n_agents) for _ in range(n_agents)]
		self.batch_size = BATCHSIZE
		self.buffer_size = BUFFERSIZE

		self.memory = ReplayBuffer(self.buffer_size , self.batch_size, self.random_seed, self.device, self.n_agents)
		self.agents_rewards = [deque(maxlen=100) for _ in range(n_agents)]

		self.cheater = cheater

		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay

		self.debug = debug

		self.update_counter = 0

	def reset(self):
		for agent in self.agents:
			agent.noise.reset()

	def get_actors(self):
		lActors = [agent.actor_local for agent in self.agents]
		tActors = [agent.actor_target for agent in self.agents]
		return lActors, tActors

	def act(self, obs_all, noise = True, target = False):
		#actions = []
		#for agent, obs in zip(self.agents, obs_all):
		#	tempAction = agent.act(obs, noise, target)
		#	actions.append(tempAction)
		if noise:
			actions = [agent.act(obs, 1., False) for agent, obs in zip(self.agents, obs_all)]
		else:
			actions = [agent.act(obs, 0., False) for agent, obs in zip(self.agents, obs_all)]
		#print(obs_all.shape)
		#actions = [agent.act(obs_all, noise, target) for agent in self.agents]
		return np.array(actions).reshape(1, -1)
	def extract_state(self, states, i):
		return states[:,i,:].reshape(self.batch_size, -1).float().to(self.device)

	def step(self, state, action, reward, next_state, done, timestep):
		self.memory.add(state.reshape(1, -1), action, reward, next_state.reshape(1, -1), done)
		self.update_counter += 1
		for i in range(self.n_agents):
			self.agents_rewards[i].append(reward[i])
		if len(self.memory) > self.batch_size and self.update_counter % self.learnSteps == 0:
			for _ in range(self.numUpdates):
				experiences = self.memory.sample()
				self.update(experiences, self.gamma)

	def update(self, experiences, n_updates = 3):
		states, actions, rewards, next_states, dones = experiences
		#states_all = states.reshape(-1, self.n_agents * self.state_size)
		#next_states_all = next_states.reshape(-1, self.n_agents * self.state_size)
		target_actions = []
		current_actions = []
		for i, agent in enumerate(self.agents):
			agent_index = torch.tensor([i]).to(self.device)
			next_states_agent = next_states.reshape(-1, 2, 24).index_select(1, agent_index).squeeze(1)
			states_agent = states.reshape(-1, 2, 24).index_select(1, agent_index).squeeze(1)
			current_actions.append(agent.actor_local(states_agent))
			with torch.no_grad():
				target_actions.append(agent.actor_target(next_states_agent))

		target_actions = torch.cat(target_actions, dim = 1)

		for i, agent in enumerate(self.agents):
			agent_index = torch.tensor([i]).to(self.device)
			###### Critic Update
			with torch.no_grad():
				Q_targets_next = agent.critic_target(next_states, target_actions)			
			Q_targets = rewards.index_select(1, agent_index) + (self.gamma * Q_targets_next * (1 - dones.index_select(1, agent_index)))
			Q_expected = agent.critic_local(states, actions)
			critic_loss = F.mse_loss(Q_expected, Q_targets)
			agent.critic_optimizer.zero_grad()
			critic_loss.backward()
			#torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
			agent.critic_optimizer.step()
			###### Actor update
			#actions_pred = [self.agents[j].actor_local(states[j,:,:]) if j == i else self.agents[j].actor_local(states[j,:,:]).detach() for j in range(self.n_agents)]
			actions_pred = [actions if j == i else actions.detach() for j, actions in enumerate(current_actions)]
			actions_pred = torch.cat(actions_pred, dim = 1).to(self.device)
			actor_loss = -agent.critic_local(states, actions_pred).mean()
			agent.actor_optimizer.zero_grad()
			actor_loss.backward()
			agent.actor_optimizer.step()
			
			agent.update(agent.critic_local, agent.critic_target, soft = True)
			agent.update(agent.actor_local, agent.actor_target, soft = True)

