#import torch
from collections import namedtuple, deque
import random
import numpy as np
import torch

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device, n_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
        self.n_agents = n_agents

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PRB:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device, alpha = 0.5, n_agents = 2):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.probabilities = deque(maxlen=buffer_size)
        self.tdErrors = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.device = device
        self.n_agents = n_agents
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        max_error = 1
        self.tdErrors.append(max_error)
        self.probabilities.append(0)
    
    def update_probabilities(self):
        priorities = np.power(np.asarray(self.tdErrors), self.alpha)
        priorities_sum = np.sum(priorities)
        priorities = priorities / priorities_sum
        self.probabilities = deque(priorities, maxlen=self.buffer_size)
    
    def update_tdErrors(self, Idx, errors):
        for Id, error in zip(Idx, errors):
            self.tdErrors[Id] = error[0]

    def sample(self, beta):
        """Randomly sample a batch of experiences from memory."""
        Idx = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p = self.probabilities)
        experiences = []
        w = []
        for Id in Idx:
            experiences.append(self.memory[Id])
            w.append(self.probabilities[Id])
        
        tot_states = torch.from_numpy(np.vstack([np.hstack(e.state) for e in experiences if e is not None])).float().to(self.device)
        tot_next_states = torch.from_numpy(np.vstack([np.hstack(e.next_state) for e in experiences if e is not None])).float().to(self.device)
        tot_actions = torch.from_numpy(np.vstack([np.hstack(e.action) for e in experiences if e is not None])).float().to(self.device)
        states = []
        actions = []
        dones = []
        rewards = []
        next_states = []
        for a in range(self.n_agents):
            states.append([e.state[a] for e in experiences if e is not None])
            next_states.append([e.next_state[a] for e in experiences if e is not None])
            actions.append([e.action[a] for e in experiences if e is not None])
            dones.append([e.done[a] for e in experiences if e is not None])
            rewards.append([e.reward[a] for e in experiences if e is not None])


        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        
        w = np.power(np.asarray(w) * self.buffer_size, -beta)
        w = w / np.max(w)
        
        return (states, tot_states, actions, tot_actions, rewards, next_states, tot_next_states, dones, Idx, w)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)