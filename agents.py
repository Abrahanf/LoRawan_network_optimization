import numpy as np
import random
from parameters import *

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Agente Q-Learning
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, node_id, avg_rssi, q_table_dict):                       # Define el constructor de Q-learning. Se ejecuta una sola vez en cada nodo
        self.node_id = node_id                                                 # Guarda el ID del nodo para depuración en la variable node_id
        self.Q = q_table_dict                                                  # Carga o creo la tabla Q-learning en la variable Q
        self.actions = [(sf, tp) for sf in SF_RANGE for tp in TX_POWERS]       # Define la lista de todas las acciones posibles que el agente puede tomar en actions
        self.rssi_bin = self.discretize_rssi(avg_rssi)                         # Guarda el valor escalado de la distancia entre valores de 0 a 4 en la varaible dist_bin
        self.alpha = ALPHA                                                     # Copia el valor de ALPHA en la variable alpha
        self.gamma = GAMMA                                                     # Copia el valor de GAMMA en la variable gamma
        self.epsilon = EPSILON_START                                           # Copia el valor de EPSILON_START en la variable espsilon

    def discretize_rssi(self, rssi):
        if rssi > -100: return 0                                               # Zona 0: Señal fortísima (Cerca del GW) -> SF7 con potencia mínima
        if rssi > -120: return 1                                               # Zona 1: Señal muy buena -> SF7 seguro
        if rssi > -123: return 2                                               # Zona 2: Límite de SF7 (-123 dBm) -> Quizás SF8
        if rssi > -126: return 3                                               # Zona 3: Límite de SF8 (-126 dBm) -> Quizás SF9
        if rssi > -129: return 4                                               # Zona 4: Límite de SF9 (-129 dBm) -> Quizás SF10
        if rssi > -132: return 5                                               # Zona 5: Límite de SF10 (-132 dBm) -> Quizás SF11
        if rssi > -134.5: return 6                                             # Zona 6: Límite de SF11 (-134.5 dBm) -> SF12 Obligatorio
        return 7

    def discretize_snr(self, snr):                                             # Convierte el SNR en valores de 0 a 4
        if snr is None: return 0                                               # Si el SNR es muy bajo se clasifica como categoría 0
        if snr < -15: return 1                                                 # Si el SNR es menor a -15 dBm se clasifica como categoría 1
        if snr < -5: return 2                                                  # Si el SNR es menor a -5 dBm se clasifica como categoría 2
        if snr < 5: return 3                                                   # Si el SNR es menor a 5 dBm se clasifica como categoría 3
        return 4                                                               # Si el SNR es mayor a 5 dBm se clasifica como categoría 4

    def get_state_key(self, congestion_level):
        return (self.rssi_bin, congestion_level)                              # Devuelve el estado completo de cuatro varaibles del agente Q-learning

    def choose_action_idx(self, state_key):                                   # Función de decisión (Epsilon-Greedy)
        if state_key not in self.Q:                                           # Si el estado "state_key" es nuevo... 
            self.Q[state_key] = np.zeros(len(self.actions))                   # lo crea y lo inicializa con una lista de ceros
        if random.random() < self.epsilon:                                    # ocurre la exploración de acuerdo al valor de epsilon
            return random.randint(0, len(self.actions) - 1)                   # Devuelve una acción totalmente al azar si el valor es menor a epsilon
        else:                                     
            qvals = self.Q[state_key]                                         # Si el valor el mayor a epsilon, se optine la lista de valores de la Q-table para el estado actual
            max_idxs = np.flatnonzero(qvals == qvals.max())                   # Encuentra el índice de la mejor acción
            return int(np.random.choice(max_idxs))                            # Devuelve el índice de la mejor acción conocida. Si hubo un empate, elige una al azar entre las mejores.

    def update_q(self, state_key, action_idx, reward, next_state_key):        # Función de aprendizaje (la Ecuación de Bellman): Q(s, a) <- Q(s, a) + a*[R + y*max_a'(Q(s', a'))-Q(s, a)]
        if state_key not in self.Q:                                           # Si el estado no está en la  Q-table...
            self.Q[state_key] = np.zeros(len(self.actions))                   # Crea uno completado de ceros
        if next_state_key not in self.Q:                                      # Si el siguiente estado no está en la Q-table...
            self.Q[next_state_key] = np.zeros(len(self.actions))              # Crea uno completado de ceros

        old_value = self.Q[state_key][action_idx]                             # Obtiene el valor antiguo de la Q-Table para el estado actual: Q(s, a)
        future_optimal_value = np.max(self.Q[next_state_key])                 # Calcula el valor futuro con el valor del siguiente estado: max_a'(Q(s', a'))
        new_value = old_value + self.alpha * (                                # Actialización del aprendizaje: Q(s. a) + a*[TD Error]
            reward + self.gamma * future_optimal_value - old_value            # Nueva estimación: (R + y*max-a'(Q(s', a'))). Error de predicción: -Q(s, a). Todo junto es el error de diferencia temporarl (TD Error)
        )
        self.Q[state_key][action_idx] = new_value                             # Guarda el nuevo conocimiento en la Q-Table
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)                 # Reduce la "curiosidad" (épsilon)

