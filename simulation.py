import simpy
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from parameters import *
from entities import *
from utils import *
from central_dqn import CentralizedAgent

# -------------------------------------------------------------------
# 🏁 7. Función Principal de Simulación
# -------------------------------------------------------------------

def run_simulation(node_count, simulation_mode, sim_duration_ms, results_dir, qtable_path):
    """
    Ejecuta una simulación completa con una configuración específica
    y devuelve el PDR global final.
    """
    
    print(f"*** Iniciando simulación: {node_count} Nodos, Modo: {simulation_mode.upper()} ***")
    
    # Inicializar cerebro central SI es necesario
    global_brain = None

    if simulation_mode == "dqn_train":
        # Modo Entrenamiento: Crea agente nuevo, entrena y guarda al final
        global_brain = CentralizedAgent(training_mode=True)
        # Usamos "dqn" internamente para la lógica de los nodos
        internal_mode = "dqn" 
        
    elif simulation_mode == "dqn_inference":
        # Modo Uso: Crea agente, CARGA el archivo y NO entrena
        global_brain = CentralizedAgent(training_mode=False)
        try:
            global_brain.load_model(qtable_path) # qtable_path trae la ruta del .pth
        except:
            print("Error: No se encontró el modelo entrenado. Ejecuta train_dqn.py primero.")
            return 0
        internal_mode = "dqn"
        
    elif simulation_mode == "dqn":
        # Modo antiguo (Entrena desde cero y borra al final)
        global_brain = CentralizedAgent(training_mode=True)
        internal_mode = "dqn"
        
    else:
        internal_mode = simulation_mode

    suffix = f"_{simulation_mode}_{node_count}n"
    EVOL_FILE = os.path.join("results", f"evolucion_pdr{suffix}.png")
    TOPO_SF_FILE = os.path.join("results",f"topologia_sf_final{suffix}.png")
    SUMMARY_FILE = os.path.join("results",f"resumen_nodos{suffix}.csv")
    

    q_all_tables = {}
    if simulation_mode == "qlearning":
        q_all_tables = load_q_tables(qtable_path)
    
    env = simpy.Environment()
    
    gateways = []
    for i, (x, y) in enumerate(GW_POSITIONS):
        gateways.append(Gateway(env, x, y, id=i))
    print(f"Creados {len(gateways)} gateways.")
    
    nodes = []
    print(f"Creando {node_count} nodos (distribución espiral)...")
    
    topo_data = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    
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
        elif simulation_mode == "adr": 
            nodes.append(NodeADR(env, gateways, node_id=i, x=x, y=y))
        elif internal_mode == "dqn":
            nodes.append(NodeDQN(env, gateways, i, x, y, global_brain))

    monitor = DataMonitor(env, gateways, nodes)

    print(f"Ejecutando simulación por {sim_duration_ms / 1000 / 60:.1f} minutos...")
    env.run(until=sim_duration_ms)
    
    if simulation_mode == "qlearning":
        save_q_tables(qtable_path, nodes)
    
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
    print("Guardando datos finales de topología y SF...")
    topo_data = []
    for n in nodes:
        final_sf = n.current_sf if hasattr(n, 'current_sf') else None
        
        topo_data.append({
            'id': n.id,
            'x': n.x,
            'y': n.y,
            'final_sf': final_sf,
            'dist_closest_gw': n.closest_gw_dist,
            'packets_sent': n.packets_sent,
            'packets_success': n.packets_success
        })
    
    topo_filename = os.path.join(results_dir, f"topologia_final_{simulation_mode}_{node_count}n.csv")
    
    try:
        df_topo = pd.DataFrame(topo_data)
        df_topo.to_csv(topo_filename, index=False)
        print(f"Topología final guardada en: {topo_filename}")
    except Exception as e:
        print(f"Error al guardar topología CSV: {e}")

    print(f"--- Fin simulación {node_count} nodos. PDR Final: {global_pdr:.2f}% ---")

    # AL FINAL: Guardar si estábamos entrenando
    if simulation_mode == "dqn_train":
        global_brain.save_model(qtable_path)

    return global_pdr
