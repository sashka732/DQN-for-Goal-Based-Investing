import gym
import QwimGym.envs.qwim_env_dir.qwim_env as qwim_env
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
from itertools import count
import numpy as np
import datetime as dt
env = qwim_env.QwimEnv(cash = 10000000, alpha_init = .5)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """saves a transition tuple"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def two_layer_net(input_dim, output_dim,
                      hidden_layer1_size=256,
                      hidden_layer2_size=256):
        """
        Generate a fully-connected two-layer network for quick use.
        """

        net = nn.Sequential(
            nn.Linear(input_dim, hidden_layer1_size),
            nn.ReLU(),
            nn.Linear(hidden_layer1_size, hidden_layer2_size),
            nn.ReLU(),
            nn.Linear(hidden_layer2_size, output_dim)
        ).double()
        return net



BATCH_SIZE = 30

GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

num_actions = env.action_space.n
policy_net = two_layer_net(input_dim = env.observation_space.shape[1],output_dim = num_actions)
target_net = two_layer_net(input_dim = env.observation_space.shape[1],output_dim = num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)

steps_done = 0

def select_action(state):
    state = torch.FloatTensor(state)
    dim = state.dim()
    global steps_done
    sample = np.random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.double()).max(0)[1].view(1, 1)
    else:
        return torch.tensor([[np.random.randint(num_actions)]], dtype=torch.long)


episode_durations = []

def optimze_model():
    if len(memory.memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([torch.tensor(s) for s in batch.next_state
                                       if s is not None]).view(-1,3).double()
    state_batch = torch.cat([torch.tensor(s) for s in batch.state]).view(-1, 3)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


num_episodes = 50

for i in range(num_episodes):
    env.reset()
    state = env.state
    for t in count():
        action = select_action(state)
        next_state, reward, done, info = env.step(action.item())
        reward = torch.tensor([reward])
        memory.push(state, action, next_state, reward)
        state = next_state
        loss = optimze_model()
        print(f"t = {t}, state = {state}, loss = {loss}")
        if done:
            episode_durations.append(t + 1)
            break

    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()

print(target_net.state_dict())