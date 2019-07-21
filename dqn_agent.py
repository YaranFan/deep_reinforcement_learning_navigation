import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork

class Agent():
    '''
    An agent that interacts with and learns from the environment.
    '''

    def __init__(self, state_size, action_size, seed=1, buffer_size=100000, batch_size=64, gamma=0.99, tau=0.001, learning_rate=0.0005, update_step=4, device='cpu'):
        # Parameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_step = update_step
        self.device = device
        self.t_step = 0

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed=seed, device=self.device)
    
    def step(self, state, action, reward, next_state, done):
        '''
        Take a step forward and learn
        '''
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = self.t_step + 1
        if ((self.t_step + 1) % self.update_step)==0 and len(self.memory)>self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, double_dqn=True)

    def act(self, state, eps=0.0):
        '''
        Returns actions for given state as per current policy.
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(random.choice(np.arange(self.action_size)))

    def learn(self, experiences, double_dqn=True):
        '''
        Update value parameters using given batch of experience tuples
        '''
        # Update local network
        states, actions, rewards, next_states, dones = experiences
        if double_dqn:
            next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)                     

    def soft_update(self, local_model, target_model):
        '''
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    '''
    Fixed-size buffer to store experience tuples.
    '''

    def __init__(self, action_size, buffer_size=100000, batch_size=64, seed=1, device='cpu'):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        '''
        Add a new experience to memory.
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        '''
        Randomly sample a batch of experiences from memory.
        '''
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''
        Return the current size of internal memory.
        '''
        return len(self.memory)