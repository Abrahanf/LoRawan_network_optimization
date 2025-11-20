import os
import random
import numpy as np
from parameters import *
from simulation import run_simulation
from central_dqn import CentralizedAgent

if __name__ == "__main__":
    print("--- ENTRENANDO CEREBRO DQN ---")
    
    RESULTS_DIR = "results"
    MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_trained_model.pth")
    
    # 1. Configuración de Entrenamiento
    # Usamos una densidad media (ej. 500 nodos) para que aprenda de congestión
    TRAIN_NODES = 500 
    TRAIN_DURATION = 100000 * 1000 # Tiempo suficiente para converger
    
    # 2. Modificamos simulation.py ligeramente para aceptar un cerebro externo
    # (Necesitarás hacer un pequeño cambio en simulation.py, ver Paso 3 abajo)
    
    # Aquí pasamos 'training_mode=True'
    brain = CentralizedAgent(training_mode=True)
    
    # Corremos la simulación pasando este cerebro
    # Nota: Tendrás que modificar run_simulation para aceptar un cerebro pre-creado
    # O, simplemente haz que run_simulation cree el agente y lo devuelva.
    
    # --- TRUCO RÁPIDO ---
    # Como run_simulation instancia el agente adentro, vamos a modificar simulation.py
    # para que guarde el modelo al final si es modo DQN.
    
    run_simulation(node_count=TRAIN_NODES, 
                   simulation_mode='dqn_train', # Usaremos un modo especial
                   sim_duration_ms=TRAIN_DURATION,
                   results_dir=RESULTS_DIR,
                   qtable_path=MODEL_PATH) # Usamos este parámetro para pasar la ruta del modelo
                   
    print("¡Entrenamiento completado!")