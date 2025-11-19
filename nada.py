import simpy                             # motor de simulación de "eventos discretos".
import random                            # crea valores aleatoreos 
import numpy as np                       # Nos permite hacer arreglos (vectores y matrices)
import math                              # Para operaciones matemáticas básicas de un solo número
import pickle                            # permite guardar el progreso de las tablas de Q-learning
from collections import defaultdict      # Diccionario que automáticamente crea una entrada con un valor predeterminado la primera vez que intentas acceder a una clave que no existe.
import matplotlib.pyplot as plt          # Para crear las imágenes .png
import pandas as pd                      # para guardar datos en .csv
import os                                # Es el sistema operativo. se usa para si exixte un archivo en la memoria
import argparse                          # analizador de argumentos. para ejecutar el script con diferentes opciones

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Parámetros de la Simulación
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parámetros Físicos
SF_RANGE = [7, 8, 9, 10, 11, 12]                  # Spreading factor
TX_POWERS = [2, 5, 8, 12, 14, 17, 20]             # Potencia de transmisión en dBm
PAYLOAD_BYTES = 20                                # Tamano de los datos útiles en cada paquete
PREAMBLE_SYMBOLS = 8                              # Estandar de LoRa. Sirve para sincronizar el nodo con el GW (preámbulo). Afecta en el cálculo del ToA
BANDWIDTH_HZ = 125000                             # Ancho de banda del canal en 125 khz
FREQ_MHZ = 915.0                                  # Frecuencia central de operación. 915 MHz para Perú
C = 3e8                                           # Velocidad de la luz en m/s

# Entorno Físico
NOISE_FLOOR_DBM = -114                           # Piso de ruido del receptor en el GW
PATH_LOSS_EXPONENT = 3.0                         # Exponente de atenuación del modelo de ptopagación
SHADOWING_SIGMA = 4.0                            # Desviación estándar del shadowing. Añade aleatoriedad realista a la pérdida de señal.

# Parámetros MAC
PACKET_INTERVAL_MS = 60 * 1000                   # El tiempo promedio que un nodo espera entre el envío de un paquete y el siguiente
PACKET_JITTER_MS = 10 * 1000                     # Jitter generado. Agrega una variación aleatoria. Evita que los nodos transmitan en patrones sincronizados.

# Simulación
N_NODES = 1000                                   # Número total de nodos en la simulación
MAX_RADIUS_M = 4000                              # El radio del círculo donde se distribuyen los nodos
SIM_TIME_MS = 80000 * 1000                       # Tiempo total que dura la simulación
N_CHANNELS = 1                                   # Número de canales de frecuencia disponible

# Definición de Gateways
GW_POSITIONS = [
    (0, 0)                                       # Cantidad y ubicación de GW en la simulación
    #(-2000,-2000 ),
    #(2000, -2000),
    #(-2000, 2000),
    #(2000, 2000)
]

# Q-Learning
ALPHA = 0.2                                      # Tasa de aprendizaje de Q-learning. Nos dice que tanto confía del resultado actual
GAMMA = 0.9                                      # Factor de descuento de Q-learning. Nos dice que tánto valora las recompensas futuras
EPSILON_START = 0.8                              # Valor incial de Epsilon. Indica el nivel de aleatoriedad con lo que se toman las deciciones
EPS_DECAY = 0.9977                               # Velocidad a la que el agente pierde su aleatoriedad (epsilon) y empieza a confiar de su experiencia
EPS_MIN = 0.05                                   # Valor mínimo que puede tomar epsilon

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Funciones Físicas y del Entorno
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def sensitivity_dbm(sf):                                                        # Define la sensibilidad mínima de recepción del gateway para cada Spreading Factor (SF).
    return {7: -123, 8: -126, 9: -129, 10: -132, 11: -134.5, 12: -137}[sf]      # Diccionario que actúa como una tabla de consulta. Si le pides

def log_distance_pl_db(distance_m, freq_mhz, n):                                # Define la función de pérdida de la señal. (Log distance path loss)
    d = max(distance_m, 1.0)                                                    # Asegura que la distancia nunca sea cero para evitar error matemático de -inf. (min 1m)
    freq_hz = freq_mhz * 1e6                                                    # Convierte la frecuencia de MHz a hZ
    fspl_1m = 20.0 * math.log10(4.0 * math.pi * freq_hz / C)                    # Pédida en espacio libre a d_0 = 1m: FSPL = PL(d_0) = 20log(d*f*4*pi/c)
    return fspl_1m + 10.0 * n * math.log10(d / 1.0)                             # Fórmula de Log-Distance Path Loss: PL = PL(d_0) + 10*n*log(d/d_0)

def time_on_air_ms(sf, payload_bytes):                                          # Define cuántos milisegundos un paquete está en el aire
    time_symbol_ms = (2**sf) / (BANDWIDTH_HZ / 1000.0)                          # Calcula cuánto dura un solo símbolo LoRa en ms: T_s = (2^SF)/BW -> T_ms = T_s*1000
    payload_symbols = 8 + max(0, math.ceil(                                     # Calcular cuántos símbolos se necesitan para enviar tus payload_bytes:
        (8 * payload_bytes - 4 * sf + 28 + 16) / (4 * sf)                       # Formula real: N_simbolos = 8 + max([(8*PL-4*SF+28+16*CRC-20*H)/(4*(SF-2*DE))]*(CR+4), 0)
    ) * 5)                                                                      # Valores usados para simplificar la fórmula: CR = 1 (4/5); DE = 0; H = 0; CRC = 1;  
    total_symbols = PREAMBLE_SYMBOLS + 4.25 + payload_symbols                   # Calcula el total de simbolos sumando el preámbulo, simbolos de los datos y otros simbolos
    return time_symbol_ms * total_symbols                                       # Calcula el ToA final multiplicando la cantidad de simbolos por el tiempo que dura cada simbolo

TOA_MAP_MS = {sf: time_on_air_ms(sf, PAYLOAD_BYTES) for sf in SF_RANGE}         # Crea un diccionario de consulta para el ToA de todos los SF {7, 8, 9, 10, 11, 12}
MIN_TOA_MS = TOA_MAP_MS[min(SF_RANGE)]                                          # Busca en el mapa el ToA más rápido (el de SF7)      
MAX_TOA_MS = TOA_MAP_MS[max(SF_RANGE)]                                          # Busca en el mapa el ToA más lento (el de SF12)
MIN_TP = min(TX_POWERS)                                                         # Busca la potencia más baja (2 dBm)
MAX_TP = max(TX_POWERS)                                                         # Busca la potencia más baja (20 dBm)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. El Agente: Clase de Q-Learning
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, node_id, closest_gw_distance, q_table_dict):            # Define el constructor de Q-learning. Se ejecuta una sola vez en cada nodo
        self.node_id = node_id                                                 # Guarda el ID del nodo para depuración en la variable node_id
        self.Q = q_table_dict                                                  # Carga o creo la tabla Q-learning en la variable Q
        self.actions = [(sf, tp) for sf in SF_RANGE for tp in TX_POWERS]       # Define la lista de todas las acciones posibles que el agente puede tomar en actions
        self.dist_bin = self.discretize_distance(closest_gw_distance)          # Guarda el valor escalado de la distancia entre valores de 0 a 4 en la varaible dist_bin
        self.alpha = ALPHA                                                     # Copia el valor de ALPHA en la variable alpha
        self.gamma = GAMMA                                                     # Copia el valor de GAMMA en la variable gamma
        self.epsilon = EPSILON_START                                           # Copia el valor de EPSILON_START en la variable espsilon

    def discretize_distance(self, distance_m):                                 # Convierte distancias en valores de 0 a 4
        bin_size = MAX_RADIUS_M / 5.0                                          # Divide todo el espacio en 5
        return min(4, int(distance_m / bin_size))                              # Divide la distancia del nodo con bin_size y lo pasa a entero para obtener valores discretos de 0 a 4

    def discretize_snr(self, snr):                                             # Convierte el SNR en valores de 0 a 4
        if snr is None: return 0                                               # Si el SNR es muy bajo se clasifica como categoría 0
        if snr < -15: return 1                                                 # Si el SNR es menor a -15 dBm se clasifica como categoría 1
        if snr < -5: return 2                                                  # Si el SNR es menor a -5 dBm se clasifica como categoría 2
        if snr < 5: return 3                                                   # Si el SNR es menor a 5 dBm se clasifica como categoría 3
        return 4                                                               # Si el SNR es mayor a 5 dBm se clasifica como categoría 4

    def get_state_key(self, congestion_level):
        """El estado solo depende de propiedades físicas estables: 
           dónde estoy (dist_bin) y qué tan llena está la red (congestion_level)."""
        return (self.dist_bin, congestion_level)      # Devuelve el estado completo de cuatro varaibles del agente Q-learning
    
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

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. El Gateway
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Gateway:
    def __init__(self, env, x, y, id):                                        # Se ejecuta al crear un nuevo gateway
        self.env = env                                                        # Almacena el entorno de simpy para poder programar eventos
        self.x = x                                                            # Almacena la posición de la cooredna
        self.y = y
        self.id = id
        self.transmissions_in_air = []
        self.packets_received = 0
        self.packets_collided = 0
        self.packets_failed_sensitivity = 0
        
        self.congestion_check_interval = 10000
        self.packets_in_interval = 0
        self.collisions_in_interval = 0
        self.current_congestion_level = 0
        self.congestion_process = env.process(self.update_congestion_level())

    def record_transmission(self, packet):
        self.transmissions_in_air.append(packet)

    def check_packet_success(self, my_packet, distance_to_this_gw):
        self.packets_in_interval += 1
        sf = my_packet['sf']
        sensitivity_req = sensitivity_dbm(sf)
        my_start = my_packet['tx_start_time']
        my_end = my_packet['tx_end_time']
        
        path_loss_db = log_distance_pl_db(distance_to_this_gw, FREQ_MHZ, PATH_LOSS_EXPONENT)
        shadowing_db = random.gauss(0.0, SHADOWING_SIGMA)
        received_power_dbm = my_packet['tp'] - path_loss_db + shadowing_db
        snr = received_power_dbm - NOISE_FLOOR_DBM
        
        try:
            self.transmissions_in_air.remove(my_packet)
        except ValueError:
            pass

        # --- ¡¡NUEVO!! Lógica de Colisión Multi-Canal ---
        interferers = 0
        my_channel = my_packet['channel'] # Obtener el canal del paquete
        
        for other in self.transmissions_in_air:
            # Colisión solo si SF, Canal y Tiempo coinciden
            if other['sf'] != sf: continue
            if other['channel'] != my_channel: continue # <-- NUEVA COMPROBACIÓN
            
            if (my_start < other['tx_end_time']) and (other['tx_start_time'] < my_end):
                interferers += 1
                
        if interferers > 0:
            self.packets_collided += 1
            self.collisions_in_interval += 1
            return "FAILED_COLLISION", snr

        if received_power_dbm < sensitivity_req:
            self.packets_failed_sensitivity += 1
            return "FAILED_SENSITIVITY", snr

        self.packets_received += 1
        return "SUCCESS", snr
    
    def update_congestion_level(self):
        while True:
            yield self.env.timeout(self.congestion_check_interval)
            if self.packets_in_interval == 0:
                collision_rate = 0.0
            else:
                collision_rate = self.collisions_in_interval / self.packets_in_interval
            
            if collision_rate < 0.1: self.current_congestion_level = 0
            elif collision_rate < 0.3: self.current_congestion_level = 1
            else: self.current_congestion_level = 2
            
            self.packets_in_interval = 0
            self.collisions_in_interval = 0

# -------------------------------------------------------------------
# 5. El OBSERVADOR (Monitor de Datos)
# -------------------------------------------------------------------
class DataMonitor:
    def __init__(self, env, gateways, nodes):
        self.env = env
        self.gateways = gateways
        self.nodes = nodes
        self.sample_interval_ms = 30 * 1000
        self.timestamps_sec = []
        self.pdr_log = []
        self.process = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.sample_interval_ms)
            total_sent = sum(n.packets_sent for n in self.nodes)
            total_recv = sum(n.packets_success for n in self.nodes)
            pdr = (total_recv / total_sent) * 100.0 if total_sent > 0 else 0.0
            self.timestamps_sec.append(self.env.now / 1000.0)
            self.pdr_log.append(pdr)

# -------------------------------------------------------------------
# 6. El Nodo (El "Cuerpo" del Agente Q-Learning)
# -------------------------------------------------------------------
class Node:
    def __init__(self, env, gateways, node_id, x, y, q_table_dict):
        self.env = env
        self.gateways = gateways
        self.id = node_id
        self.x, self.y = x, y

        self.closest_gw_dist = float('inf')
        self.closest_gw = None
        for gw in gateways:
            dist = max(1.0, math.sqrt((self.x - gw.x)**2 + (self.y - gw.y)**2))
            if dist < self.closest_gw_dist:
                self.closest_gw_dist = dist
                self.closest_gw = gw
        
        self.agent = QLearningAgent(node_id, self.closest_gw_dist, q_table_dict)
        
        self.last_snr = None
        self.last_result = None
        self.current_sf = None
        self.current_tp = None
        self.packets_sent = 0
        self.packets_success = 0
        self.process = env.process(self.run_aloha())

    def run_aloha(self):
        yield self.env.timeout(random.uniform(0, PACKET_INTERVAL_MS))
        
        while True:
            # 1. OBTENER ESTADO
            current_congestion = self.closest_gw.current_congestion_level
            current_state_key = self.agent.get_state_key(
                current_congestion
            )

            # 2. DECIDIR ACCIÓN
            action_idx = self.agent.choose_action_idx(current_state_key)
            self.current_sf, self.current_tp = self.agent.actions[action_idx]
            
            # --- ¡¡NUEVO!! Simular Salto de Frecuencia ---
            current_channel = random.randint(0, N_CHANNELS - 1)

            # 3. EJECUTAR ACCIÓN
            self.packets_sent += 1
            toa = TOA_MAP_MS[self.current_sf]
            tx_start_time = self.env.now
            tx_end_time = tx_start_time + toa
            
            packet = {
                'node_id': self.id, 'sf': self.current_sf, 'tp': self.current_tp,
                'channel': current_channel, # <-- Añadir canal al paquete
                'tx_start_time': tx_start_time, 'tx_end_time': tx_end_time,
                'node_x': self.x, 'node_y': self.y
            }
            
            for gw in self.gateways:
                gw.record_transmission(packet)
                
            yield self.env.timeout(toa)
            
            # 4. OBSERVAR RESULTADO
            best_result = "FAILED_SENSITIVITY"
            best_snr = -999.0

            for gw in self.gateways:
                dist_to_gw = max(1.0, math.sqrt((self.x - gw.x)**2 + (self.y - gw.y)**2))
                result, snr = gw.check_packet_success(packet, dist_to_gw)
                
                if result == "SUCCESS":
                    best_result = "SUCCESS"
                    if snr > best_snr:
                        best_snr = snr
                elif result == "FAILED_COLLISION" and best_result != "SUCCESS":
                    best_result = "FAILED_COLLISION"
                elif result == "FAILED_SENSITIVITY" and best_result not in ["SUCCESS", "FAILED_COLLISION"]:
                    if snr > best_snr:
                        best_snr = snr

            # 5. CALCULAR RECOMPENSA
            result_for_reward = best_result
            reward = self.calculate_reward(result_for_reward, self.current_sf, self.current_tp)
            
            # 6. DEFINIR EL PRÓXIMO ESTADO Y APRENDER
            next_snr_state = -999.0
            if result_for_reward == "SUCCESS":
                self.packets_success += 1
                next_snr_state = best_snr
            elif result_for_reward == "FAILED_SENSITIVITY":
                next_snr_state = best_snr
            
            next_state_key = self.agent.get_state_key(
                current_congestion
            )
            
            self.agent.update_q(current_state_key, action_idx, reward, next_state_key)
            
            self.last_snr = next_snr_state
            self.last_result = result_for_reward

            # 7. ESPERAR
            interval = PACKET_INTERVAL_MS + random.uniform(-PACKET_JITTER_MS, PACKET_JITTER_MS)
            yield self.env.timeout(max(1000.0, interval - toa))

    def calculate_reward(self, result, sf, tp):
        if result == "SUCCESS":
            r_pdr = 10.0
        else:
            r_pdr = -1.0
        toa = TOA_MAP_MS[sf]
        norm_toa = (toa - MIN_TOA_MS) / (MAX_TOA_MS - MIN_TOA_MS)
        norm_tp = (tp - MIN_TP) / (MAX_TP - MIN_TP) if (MAX_TP - MIN_TP) > 0 else 0
        w_toa = 0.7
        w_tp = 0.3
        cost_energy = (w_toa * norm_toa + w_tp * norm_tp)
        return r_pdr - cost_energy

# -------------------------------------------------------------------
# 6.5. El Nodo de Comparación (ADR Nativo)
# -------------------------------------------------------------------

ADR_HISTORY_LEN = 20
ADR_MARGIN_UP = 15
ADR_MARGIN_DOWN = 5

class NodeADR:
    def __init__(self, env, gateways, node_id, x, y, **kwargs):
        self.env = env
        self.gateways = gateways
        self.id = node_id
        self.x, self.y = x, y
        self.snr_history = []
        self.current_sf = 12
        self.current_tp = MAX_TP
        self.packets_sent = 0
        self.packets_success = 0
        self.process = env.process(self.run_adr_aloha())
        self.closest_gw_dist = min(
            [max(1.0, math.sqrt((self.x - gw.x)**2 + (self.y - gw.y)**2)) for gw in gateways]
        )

    def get_required_snr(self, sf):
        return sensitivity_dbm(sf) - NOISE_FLOOR_DBM

    def get_next_adr_params(self):
        if len(self.snr_history) < ADR_HISTORY_LEN:
            return self.current_sf, self.current_tp
        valid_snrs = [s for s in self.snr_history if s is not None and s > -999]
        if not valid_snrs:
            return self.current_sf, self.current_tp
        avg_snr = np.mean(valid_snrs)
        required_snr = self.get_required_snr(self.current_sf)
        current_margin = avg_snr - required_snr
        new_sf = self.current_sf
        new_tp = self.current_tp
        if current_margin > ADR_MARGIN_UP:
            tp_idx = TX_POWERS.index(self.current_tp)
            if tp_idx > 0:
                new_tp = TX_POWERS[tp_idx - 1]
            else:
                sf_idx = SF_RANGE.index(self.current_sf)
                if sf_idx > 0:
                    new_sf = SF_RANGE[sf_idx - 1]
                    new_tp = MAX_TP 
        elif current_margin < ADR_MARGIN_DOWN:
            sf_idx = SF_RANGE.index(self.current_sf)
            if sf_idx < len(SF_RANGE) - 1:
                new_sf = SF_RANGE[sf_idx + 1]
                new_tp = MAX_TP 
        self.snr_history = []
        return new_sf, new_tp

    def run_adr_aloha(self):
        yield self.env.timeout(random.uniform(0, PACKET_INTERVAL_MS))
        
        while True:
            # 1. DECIDIR ACCIÓN
            self.current_sf, self.current_tp = self.get_next_adr_params()
            
            # --- ¡¡NUEVO!! Simular Salto de Frecuencia ---
            current_channel = random.randint(0, N_CHANNELS - 1)

            # 2. EJECUTAR ACCIÓN
            self.packets_sent += 1
            toa = TOA_MAP_MS[self.current_sf]
            tx_start_time = self.env.now
            tx_end_time = tx_start_time + toa
            packet = {
                'node_id': self.id, 'sf': self.current_sf, 'tp': self.current_tp,
                'channel': current_channel, # <-- Añadir canal al paquete
                'tx_start_time': tx_start_time, 'tx_end_time': tx_end_time,
                'node_x': self.x, 'node_y': self.y
            }
            
            for gw in self.gateways:
                gw.record_transmission(packet)
                
            yield self.env.timeout(toa)
            
            # 3. OBSERVAR RESULTADO
            best_result = "FAILED_SENSITIVITY"
            best_snr = -999.0
            for gw in self.gateways:
                dist_to_gw = max(1.0, math.sqrt((self.x - gw.x)**2 + (self.y - gw.y)**2))
                result, snr = gw.check_packet_success(packet, dist_to_gw)
                if result == "SUCCESS":
                    best_result = "SUCCESS"
                    if snr > best_snr:
                        best_snr = snr
                elif result == "FAILED_COLLISION" and best_result != "SUCCESS":
                    best_result = "FAILED_COLLISION"
                elif result == "FAILED_SENSITIVITY" and best_result not in ["SUCCESS", "FAILED_COLLISION"]:
                    if snr > best_snr:
                        best_snr = snr
            
            # 4. ACTUALIZAR HISTORIAL DE ADR
            if best_result == "SUCCESS":
                self.packets_success += 1
                self.snr_history.append(best_snr)
            elif best_result == "FAILED_SENSITIVITY":
                self.snr_history.append(best_snr)
            if len(self.snr_history) > ADR_HISTORY_LEN:
                self.snr_history.pop(0)

            # 5. ESPERAR
            interval = PACKET_INTERVAL_MS + random.uniform(-PACKET_JITTER_MS, PACKET_JITTER_MS)
            yield self.env.timeout(max(1000.0, interval - toa))

# -------------------------------------------------------------------
# 7. Script Principal (Setup y Ejecución)
# -------------------------------------------------------------------

def load_q_tables(filename):
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            q_all = {}
            for node_id, q_table in data.items():
                action_len = len(SF_RANGE) * len(TX_POWERS)
                if len(q_table.get(list(q_table.keys())[0], [])) != action_len:
                     print(f"Descartando Q-table de Nodo {node_id} (tamaño de acción no coincide)")
                     continue
                new_q = defaultdict(lambda: np.zeros(action_len))
                new_q.update(q_table)
                q_all[str(node_id)] = new_q
            print(f"Q-Tables cargadas desde {filename}")
            return q_all
    except Exception as e:
        print(f"Error cargando Q-tables: {e}. Empezando de cero.")
        return {}

def save_q_tables(filename, nodes):
    q_all_serializable = {}
    for n in nodes:
        if isinstance(n, Node):
            q_all_serializable[str(n.id)] = dict(n.agent.Q)
    with open(filename, "wb") as f:
        pickle.dump(q_all_serializable, f)
    print(f"Q-Tables guardadas en {filename}")


# -------------------------------------------------------------------
# 🏁 7. Función Principal de Simulación
# -------------------------------------------------------------------
def plot_topology(nodes, gateways, filename):
    """
    Genera un gráfico de dispersión mostrando la ubicación de los nodos,
    coloreados según su SF final, y la posición de los Gateways.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Extraer datos de los nodos
    x_coords = [n.x for n in nodes]
    y_coords = [n.y for n in nodes]
    sfs = [n.current_sf for n in nodes]
    
    # 2. Definir colores para cada SF
    # Colores: 7=Azul, 8=Cyan, 9=Verde, 10=Amarillo, 11=Naranja, 12=Rojo
    colors = {7: 'blue', 8: 'cyan', 9: 'green', 10: 'yellow', 11: 'orange', 12: 'red'}
    
    # 3. Graficar nodos por grupo de SF (para que la leyenda salga bien)
    unique_sfs = sorted(list(set(sfs)))
    for sf in unique_sfs:
        # Filtrar nodos que tienen este SF
        idx = [i for i, val in enumerate(sfs) if val == sf]
        lx = [x_coords[i] for i in idx]
        ly = [y_coords[i] for i in idx]
        
        c = colors.get(sf, 'black') # Negro si es un SF raro
        plt.scatter(lx, ly, c=c, label=f'SF {sf}', s=20, alpha=0.7, edgecolors='none')

    # 4. Graficar Gateways
    gx = [gw.x for gw in gateways]
    gy = [gw.y for gw in gateways]
    plt.scatter(gx, gy, c='black', marker='^', s=150, label='Gateway', edgecolors='white')

    # 5. Configuración del gráfico
    plt.title(f"Topología Final y Distribución de SF\n{filename}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Para que el círculo no se vea ovalado
    plt.tight_layout()
    
    # 6. Guardar
    plt.savefig(filename)
    plt.close()
    print(f"Gráfico guardado: {filename}")

def run_simulation(node_count, simulation_mode, sim_duration_ms):
    """
    Ejecuta una simulación completa con una configuración específica
    y devuelve el PDR global final.
    """
    
    print(f"*** Iniciando simulación: {node_count} Nodos, Modo: {simulation_mode.upper()} ***")
    
    # Nombres de archivo dinámicos para guardar los resultados de CADA simulación
    suffix = f"_{simulation_mode}_{node_count}n"
    EVOL_FILE = f"evolucion_pdr{suffix}.png"
    TOPO_SF_FILE = f"topologia_sf_final{suffix}.png"
    SUMMARY_FILE = f"resumen_nodos{suffix}.csv"
    QTABLE_FILE = "q_tables.pkl" # Mantenemos una Q-Table compartida

    q_all_tables = {}
    if simulation_mode == "qlearning":
        # ¡Importante! Cargamos la Q-Table para seguir aprendiendo
        q_all_tables = load_q_tables(QTABLE_FILE)
    
    env = simpy.Environment()
    
    gateways = []
    for i, (x, y) in enumerate(GW_POSITIONS):
        gateways.append(Gateway(env, x, y, id=i))
    print(f"Creados {len(gateways)} gateways.")
    
    nodes = []
    print(f"Creando {node_count} nodos (distribución espiral)...")
    
    topo_data = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
    # Usamos node_count aquí
    for i in range(node_count):
        r = np.sqrt((i + 0.5) / node_count) * MAX_RADIUS_M
        theta = i * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        node_dist = max(1.0, math.sqrt(x**2 + y**2))
        topo_data.append({"ID": i, "x": x, "y": y, "distance_m": node_dist})

        if simulation_mode == "qlearning":
            action_len = len(SF_RANGE) * len(TX_POWERS)
            node_q_table = q_all_tables.get(str(i), defaultdict(lambda: np.zeros(action_len)))
            nodes.append(Node(env, gateways, node_id=i, x=x, y=y, q_table_dict=node_q_table))
        else: # modo "adr"
            nodes.append(NodeADR(env, gateways, node_id=i, x=x, y=y))
            
    # pd.DataFrame(topo_data).to_csv(TOPO_FILE, index=False) # Opcional: guardar topología
    
    monitor = DataMonitor(env, gateways, nodes)

    print(f"Ejecutando simulación por {sim_duration_ms / 1000 / 60:.1f} minutos...")
    env.run(until=sim_duration_ms)
    
    if simulation_mode == "qlearning":
        # Guardamos la Q-Table actualizada para la siguiente simulación
        save_q_tables(QTABLE_FILE, nodes)
    
    print("\n--- Simulación Finalizada ---")
    
    total_sent = sum(n.packets_sent for n in nodes)
    total_success_de_dup = sum(n.packets_success for n in nodes)
    total_collided = sum(gw.packets_collided for gw in gateways)
    total_failed_sens = sum(gw.packets_failed_sensitivity for gw in gateways)
    global_pdr = (total_success_de_dup / total_sent) * 100.0 if total_sent > 0 else 0.0

    print("\n--- Resultados de Red (Agregados) ---")
    print(f"Paquetes Totales Enviados (Nodos):   {total_sent}")
    print(f"Paquetes Recibidos (Éxito Único):  {total_success_de_dup}")
    print(f"PDR (Packet Delivery Rate) Global: {global_pdr:.2f}%")
    
    # --- Opcional: Guardar reportes individuales ---
    # (Puedes comentar estas secciones para acelerar el bucle)
    
    # summary_data = [] # ... (código para crear summary_data) ...
    # pd.DataFrame(summary_data).sort_values("Distancia_Closest_GW (m)").to_csv(SUMMARY_FILE, index=False)
    # print(f"Resumen de nodos guardado en '{SUMMARY_FILE}'")

    # print("Generando gráfico de evolución del PDR...")
    # plt.figure(figsize=(10, 5))
    # plt.plot(monitor.timestamps_sec, monitor.pdr_log, marker='.', linestyle='-')
    # plt.title(f"Evolución PDR ({node_count} Nodos, {len(gateways)} GWs, {N_CHANNELS} Canales, {simulation_mode.upper()})")
    # plt.savefig(EVOL_FILE)
    # plt.close()
    # print(f"Gráfico de evolución guardado en '{EVOL_FILE}'")
    image_filename = f"topologia_sf_{simulation_mode}_{node_count}n.png"
    
    # Llamar a la función de ploteo
    plot_topology(nodes, gateways, image_filename)

    print(f"--- Fin simulación {node_count} nodos. PDR Final: {global_pdr:.2f}% ---")
    
    # Devuelve el resultado final
    return global_pdr

# -------------------------------------------------------------------
# 🏁 8. Script Principal (Lanzador de Pruebas de Escalabilidad)
# -------------------------------------------------------------------

if __name__ == "__main__":
    
    # --- CONFIGURACIÓN DE PRUEBAS ---
    # Define la lista de "puntos de datos" (conteo de nodos) para tu gráfico
    NODE_TEST_COUNTS = [100]
    #NODE_TEST_COUNTS = [50, 200, 500]
    # Define la duración de CADA simulación.
    # (Advertencia: 20 horas simuladas (72000*1000) por 6 pruebas tomará MUCHO tiempo real)
    # (Quizás quieras bajarlo a 1 hora (3600*1000) para una prueba más rápida)
    #SIM_DURATION_PER_RUN = 3600 * 1000 # 1 hora simulada (para pruebas rápidas)
    SIM_DURATION_PER_RUN = 80000 * 1000 # 20 horas simuladas (para resultados finales)
    
    # Listas para guardar los resultados finales
    results_qlearning = []
    results_adr = []

    # --- EJECUTAR PRUEBAS Q-LEARNING ---
    print("--- INICIANDO PRUEBAS DE ESCALABILIDAD: Q-LEARNING ---")
    # Limpiar/reiniciar la Q-Table antes de empezar
    if os.path.exists("q_tables.pkl"):
        os.remove("q_tables.pkl")
        
    for count in NODE_TEST_COUNTS:
        random.seed(0) # Reiniciar la semilla para que la simulación sea comparable
        np.random.seed(0)
        
        pdr = run_simulation(node_count=count, 
                             simulation_mode='qlearning', 
                             sim_duration_ms=SIM_DURATION_PER_RUN)
        results_qlearning.append(pdr)
        
    # --- EJECUTAR PRUEBAS ADR ---
    print("\n\n--- INICIANDO PRUEBAS DE ESCALABILIDAD: ADR ---")
    for count in NODE_TEST_COUNTS:
        random.seed(0) # Reiniciar la semilla
        np.random.seed(0)
        
        pdr = run_simulation(node_count=count, 
                             simulation_mode='adr', 
                             sim_duration_ms=SIM_DURATION_PER_RUN)
        results_adr.append(pdr)

    # --- 4. GENERAR GRÁFICO FINAL DE ESCALABILIDAD ---
    print("\n\n--- Todas las simulaciones completadas. Generando gráfico final... ---")
    
    plt.figure(figsize=(10, 6))
    plt.plot(NODE_TEST_COUNTS, results_qlearning, label="Q-Learning (Network-Aware)", marker='o', markersize=8, linewidth=2)
    plt.plot(NODE_TEST_COUNTS, results_adr, label="ADR Nativo (Link-Based)", marker='x', markersize=8, linestyle='--', linewidth=2)
    
    plt.title(f"Comparativa de Escalabilidad (PDR vs Densidad)\n({len(GW_POSITIONS)} GWs, {N_CHANNELS} Canales)")
    plt.xlabel("Número de Nodos en la Red (Densidad)")
    plt.ylabel("PDR Global (%)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100) # El PDR siempre es de 0 a 100
    
    final_graph_file = "comparativa_escalabilidad.png"
    plt.savefig(final_graph_file)
    plt.close()

    print(f"¡Éxito! Gráfico de escalabilidad guardado en: {final_graph_file}")
    
    # Imprimir resultados para copiar y pegar
    print("\nResultados (Q-Learning):", results_qlearning)
    print("Resultados (ADR):", results_adr)