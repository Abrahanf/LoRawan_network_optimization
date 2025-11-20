import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
from parameters import SF_RANGE, TX_POWERS

# Parámetros
BATCH_SIZE = 32
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
    def __init__(self, training_mode=True):
        self.actions = [(sf, tp) for sf in SF_RANGE for tp in TX_POWERS]
        self.n_actions = len(self.actions)
        self.n_states = 2 
        self.training_mode = training_mode # <-- Bandera de control

        self.policy_net = DQN(self.n_states, self.n_actions)

        if self.training_mode:
            self.target_net = DQN(self.n_states, self.n_actions)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
            self.memory = deque(maxlen=50000) 
        else:
            self.policy_net.eval() # Modo evaluación (más rápido)
        self.steps_done = 0
        self.learn_step_counter = 0
    
    def normalize_state(self, rssi, congestion):
        # Normalización robusta
        norm_rssi = (rssi + 140.0) / 100.0 
        norm_cong = float(congestion) / 2.0
        return torch.FloatTensor([norm_rssi, norm_cong])

    def select_action(self, rssi, congestion):
        if not self.training_mode:
            with torch.no_grad():
                state = self.normalize_state(rssi, congestion).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
            
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state = self.normalize_state(rssi, congestion).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.n_actions)

    def store_transition(self, rssi, cong, action, reward, next_rssi, next_cong):
        if not self.training_mode:
            return
        state = self.normalize_state(rssi, cong)
        next_state = self.normalize_state(next_rssi, next_cong)
        action_tensor = torch.LongTensor([action])
        reward_tensor = torch.FloatTensor([reward])
        
        self.memory.append((state, action_tensor, reward_tensor, next_state))

    def optimize_model(self):
        if not self.training_mode:
            return
        if len(self.memory) < BATCH_SIZE:
            return
        
        self.learn_step_counter += 1
        if self.learn_step_counter % 10 != 0:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = torch.stack([x[0] for x in batch])
        action_batch = torch.LongTensor([[x[1]] for x in batch])
        reward_batch = torch.stack([x[2] for x in batch])
        next_state_batch = torch.stack([x[3] for x in batch])
        
        # Q(s, a)
        q_eval = self.policy_net(state_batch).gather(1, action_batch)

        # Q(s', a') max - Usando Target Net
        with torch.no_grad():
            q_next = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
            q_target = reward_batch + (GAMMA * q_next)

        loss = nn.SmoothL1Loss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        if self.training_mode:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
    # --- NUEVAS FUNCIONES PARA GUARDAR Y CARGAR ---
    def save_model(self, filename="dqn_brain.pth"):
        torch.save(self.policy_net.state_dict(), filename)
        print(f"Modelo guardado en {filename}")

    def load_model(self, filename="dqn_brain.pth"):
        # Carga los pesos guardados
        self.policy_net.load_state_dict(torch.load(filename))
        self.policy_net.eval() # Pone la red en modo inferencia
        print(f"Modelo cargado desde {filename}")
