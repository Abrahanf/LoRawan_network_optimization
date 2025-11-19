import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
from parameters import SF_RANGE, TX_POWERS

# Parámetros
BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000  
LR = 0.001       

# 1. Red Neuronal 
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

class CentralizedAgent:
    def __init__(self):
        self.actions = [(sf, tp) for sf in SF_RANGE for tp in TX_POWERS]
        self.n_actions = len(self.actions)
        self.n_states = 2 

        self.policy_net = DQN(self.n_states, self.n_actions)
        self.target_net = DQN(self.n_states, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=20000) 
        self.steps_done = 0

    def select_action(self, rssi, congestion):
        norm_rssi = (rssi + 140) / 100.0 
        norm_cong = congestion / 2.0
        
        state_tensor = torch.FloatTensor([norm_rssi, norm_cong])

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(0)[1].item()
                return action_idx
        else:
            return random.randrange(self.n_actions)

    def store_transition(self, rssi, congestion, action_idx, reward, next_rssi, next_cong):
        state = torch.FloatTensor([(rssi + 140)/100.0, congestion/2.0])
        next_state = torch.FloatTensor([(next_rssi + 140)/100.0, next_cong/2.0])
        
        self.memory.append((state, action_idx, reward, next_state))

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = torch.stack([x[0] for x in batch])
        action_batch = torch.LongTensor([[x[1]] for x in batch])
        reward_batch = torch.FloatTensor([x[2] for x in batch])
        next_state_batch = torch.stack([x[3] for x in batch])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
