import math
import random
import numpy as np
import simpy
from parameters import *
from lora_physics import *
from agents import QLearningAgent
from central_dqn import *
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

        interferers = 0
        my_channel = my_packet['channel']
        
        for other in self.transmissions_in_air:
            if other['sf'] != sf: continue
            if other['channel'] != my_channel: continue
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
        
        GW_TX_POWER_DBM = 14.0                                                                          # Potencia estándar de Tx del Gateway (Downlink)
        path_loss_downlink = log_distance_pl_db(self.closest_gw_dist, FREQ_MHZ, PATH_LOSS_EXPONENT)
        avg_rssi = GW_TX_POWER_DBM - path_loss_downlink
        self.agent = QLearningAgent(node_id, avg_rssi, q_table_dict)
        
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
            current_state_key = self.agent.get_state_key(current_congestion)

            # 2. DECIDIR ACCIÓN
            action_idx = self.agent.choose_action_idx(current_state_key)
            self.current_sf, self.current_tp = self.agent.actions[action_idx]
            current_channel = random.randint(0, N_CHANNELS - 1)

            # 3. EJECUTAR ACCIÓN
            self.packets_sent += 1
            toa = TOA_MAP_MS[self.current_sf]
            tx_start_time = self.env.now
            tx_end_time = tx_start_time + toa
            
            packet = {
                'node_id': self.id, 'sf': self.current_sf, 'tp': self.current_tp,
                'channel': current_channel, 
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
            
            next_state_key = self.agent.get_state_key(current_congestion)
            
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
ADR_ACK_LIMIT = 10

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
        self.adr_failure_counter = 0
        self.process = env.process(self.run_adr_aloha())
        self.closest_gw_dist = min(
            [max(1.0, math.sqrt((self.x - gw.x)**2 + (self.y - gw.y)**2)) for gw in gateways]
        )

    def get_required_snr(self, sf):
        return sensitivity_dbm(sf) - NOISE_FLOOR_DBM

    def get_next_adr_params(self):
        
        if self.adr_failure_counter >= ADR_ACK_LIMIT:
            self.adr_failure_counter = 0 
            self.snr_history = []        
            
            current_tp_idx = TX_POWERS.index(self.current_tp)
            if current_tp_idx < len(TX_POWERS) - 1:
                new_tp = TX_POWERS[current_tp_idx + 1]
                return self.current_sf, new_tp
            
            current_sf_idx = SF_RANGE.index(self.current_sf)
            if current_sf_idx < len(SF_RANGE) - 1:
                new_sf = SF_RANGE[current_sf_idx + 1]
                return new_sf, MAX_TP 
            
            return self.current_sf, self.current_tp
        
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
        
        if new_sf != self.current_sf or new_tp != self.current_tp:
            self.snr_history = []
            
        return new_sf, new_tp

    def run_adr_aloha(self):
        yield self.env.timeout(random.uniform(0, PACKET_INTERVAL_MS))
        
        while True:
            # 1. DECIDIR ACCIÓN
            self.current_sf, self.current_tp = self.get_next_adr_params()
            
            current_channel = random.randint(0, N_CHANNELS - 1)

            # 2. EJECUTAR ACCIÓN
            self.packets_sent += 1
            toa = TOA_MAP_MS[self.current_sf]
            tx_start_time = self.env.now
            tx_end_time = tx_start_time + toa
            packet = {
                'node_id': self.id, 'sf': self.current_sf, 'tp': self.current_tp,
                'channel': current_channel,
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
            
            if best_result == "SUCCESS":
                self.packets_success += 1
                self.snr_history.append(best_snr)                                                   # Guarda SNR para optimizar
                self.adr_failure_counter = 0                                                        # Éxito: Reinicia el contador de fallos
            
            elif best_result == "FAILED_SENSITIVITY":
                self.snr_history.append(best_snr)                                                   # Guarda SNR (bajo) para optimizar
                self.adr_failure_counter += 1                                                       # Fallo: Incrementa el contador
                
            elif best_result == "FAILED_COLLISION":
                self.adr_failure_counter += 1                                                       # Fallo: Incrementa el contador
            
            if len(self.snr_history) > ADR_HISTORY_LEN:
                self.snr_history.pop(0)

            # 5. ESPERAR
            interval = PACKET_INTERVAL_MS + random.uniform(-PACKET_JITTER_MS, PACKET_JITTER_MS)
            yield self.env.timeout(max(1000.0, interval - toa))



class NodeDQN:
    def __init__(self, env, gateways, node_id, x, y, global_brain):
        self.env = env
        self.gateways = gateways
        self.id = node_id
        self.x, self.y = x, y
        self.brain = global_brain
        
        self.closest_gw_dist = min([max(1.0, math.sqrt((self.x - gw.x)**2 + (self.y - gw.y)**2)) for gw in gateways])
        path_loss = log_distance_pl_db(self.closest_gw_dist, FREQ_MHZ, PATH_LOSS_EXPONENT)
        self.avg_rssi = 14.0 - path_loss
        
        self.current_sf = 12
        self.current_tp = 14
        self.packets_sent = 0
        self.packets_success = 0
        self.process = env.process(self.run())

    def calculate_reward(self, result, sf, tp):
        if result == "SUCCESS": r_pdr = 10.0
        else: r_pdr = -1.0
        
        toa = TOA_MAP_MS[sf]
        norm_toa = (toa - MIN_TOA_MS) / (MAX_TOA_MS - MIN_TOA_MS)
        norm_tp = (tp - MIN_TP) / (MAX_TP - MIN_TP)
        cost = (0.7 * norm_toa + 0.3 * norm_tp)
        return r_pdr - cost

    def run(self):
        yield self.env.timeout(random.uniform(0, PACKET_INTERVAL_MS))
        
        while True:
            # 1. Preguntar al cerebro qué hacer
            current_congestion = self.gateways[0].current_congestion_level 
            action_idx = self.brain.select_action(self.avg_rssi, current_congestion)
            self.current_sf, self.current_tp = self.brain.actions[action_idx]
            
            # 2. Elegir canal y tiempo
            current_channel = random.randint(0, N_CHANNELS - 1)
            toa = TOA_MAP_MS[self.current_sf]
            
            packet = {
                'node_id': self.id, 'sf': self.current_sf, 'tp': self.current_tp,
                'channel': current_channel,
                'tx_start_time': self.env.now,
                'tx_end_time': self.env.now + toa,
                'node_x': self.x, 'node_y': self.y
            }
            
            # 3. Enviar
            self.packets_sent += 1
            for gw in self.gateways:
                gw.record_transmission(packet)
            yield self.env.timeout(toa)
            
            # 4. Ver resultado
            best_result = "FAILED_SENSITIVITY"
            for gw in self.gateways:
                dist = max(1.0, math.sqrt((self.x - gw.x)**2 + (self.y - gw.y)**2))
                res, _ = gw.check_packet_success(packet, dist)
                if res == "SUCCESS": best_result = "SUCCESS"
                elif res == "FAILED_COLLISION" and best_result != "SUCCESS": best_result = "FAILED_COLLISION"
            
            if best_result == "SUCCESS": self.packets_success += 1

            reward = self.calculate_reward(best_result, self.current_sf, self.current_tp)
            
            next_congestion = self.gateways[0].current_congestion_level

            # El cerebro guarda la experiencia
            self.brain.store_transition(
                self.avg_rssi, current_congestion,      # Estado actual
                action_idx, reward,             # Acción y Recompensa
                self.avg_rssi, current_congestion       # Siguiente estado
            )

            self.brain.optimize_model()
            
            self.brain.update_target_network()
            
            # 6. Esperar
            interval = PACKET_INTERVAL_MS + random.uniform(-PACKET_JITTER_MS, PACKET_JITTER_MS)
            yield self.env.timeout(max(1000.0, interval - toa))
