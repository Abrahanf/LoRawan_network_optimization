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
    TRAIN_NODES = 1000 
    TRAIN_DURATION = 100000 * 1000
    brain = CentralizedAgent(training_mode=True)
    
    run_simulation(node_count=TRAIN_NODES, 
                   simulation_mode='dqn_train',
                   sim_duration_ms=TRAIN_DURATION,
                   results_dir=RESULTS_DIR,
                   qtable_path=MODEL_PATH) 
                   
    print("¡Entrenamiento completado!")