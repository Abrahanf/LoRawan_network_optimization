import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
from parameters import SF_RANGE, TX_POWERS

# --- Hiperparámetros de la IA ---
BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000  # Decae más lento porque hay muchos nodos aprendiendo a la vez
LR = 0.001         # Tasa de aprendizaje

# 1. La Red Neuronal (El modelo matemático)
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

# 2. El Agente Central (El que toma decisiones)
class CentralizedAgent:
    def __init__(self):
        # Acciones: Todas las combinaciones posibles de (SF, TP)
        self.actions = [(sf, tp) for sf in SF_RANGE for tp in TX_POWERS]
        self.n_actions = len(self.actions)
        
        # Estado: [RSSI_Normalizado, Congestion_Level]
        # Usamos solo 2 valores para ser justos con tu Q-Learning
        self.n_states = 2 
        
        # Cerebro (Política) y Cerebro de Respaldo (Target)
        self.policy_net = DQN(self.n_states, self.n_actions)
        self.target_net = DQN(self.n_states, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=20000) # Memoria grande para guardar experiencia de todos
        self.steps_done = 0

    def select_action(self, rssi, congestion):
        # Normalizar inputs para la red neuronal (0 a 1 aprox)
        # RSSI suele ser -120 a -60. Lo movemos a 0.0 - 1.0
        norm_rssi = (rssi + 140) / 100.0 
        norm_cong = congestion / 2.0
        
        state_tensor = torch.FloatTensor([norm_rssi, norm_cong])

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                # La red decide la mejor acción
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(0)[1].item()
                return action_idx
        else:
            # Exploración aleatoria
            return random.randrange(self.n_actions)

    def store_transition(self, rssi, congestion, action_idx, reward, next_rssi, next_cong):
        # Normalizar todo antes de guardar
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

        # Q(s, a) actual
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # V(s') futuro (usando la red target para estabilidad)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Calcular error y entrenar
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        # Actualizar la red de respaldo ocasionalmente
        self.target_net.load_state_dict(self.policy_net.state_dict())