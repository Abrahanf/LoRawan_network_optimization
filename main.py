import numpy as np
import random
import os
import matplotlib.pyplot as plt
from parameters import *
from simulation import *

# -------------------------------------------------------------------------------------------------------------------------------
# 8. Script Principal (Lanzador de Pruebas de Escalabilidad)
# -------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    
    NODE_TEST_COUNTS = [10,100,200,500,1000,2000]
    SIM_DURATION_PER_RUN = 80000 * 1000               
    results_qlearning = []
    results_adr = []
    
    print("\n\n--- INICIANDO PRUEBAS DE ESCALABILIDAD: Q-LEARNING ---")              # EJECUTAR PRUEBAS Q-LEARNING
    for count in NODE_TEST_COUNTS: 
        QTABLE_PATH = os.path.join("results", "q_tables.pkl")
        if os.path.exists(QTABLE_PATH):
            os.remove(QTABLE_PATH)
        pdr = run_simulation(node_count=count, 
                             simulation_mode='qlearning', 
                             sim_duration_ms=SIM_DURATION_PER_RUN,
                             results_dir="results",  
                             qtable_path=QTABLE_PATH)   
        results_qlearning.append(pdr)
        csv_path = os.path.join("results", f"topologia_final_qlearning_{count}n.csv")
        png_path = os.path.join("results", f"topologia_sf_qlearning_{count}n.png")
        plot_sf_distribution(csv_path, png_path)

    print("\n\n--- INICIANDO PRUEBAS DE ESCALABILIDAD: ADR ---")                      # EJECUTAR PRUEBAS ADR
    for count in NODE_TEST_COUNTS:
        pdr = run_simulation(node_count=count, 
                             simulation_mode='adr', 
                             sim_duration_ms=SIM_DURATION_PER_RUN,
                            results_dir="results",  
                             qtable_path=QTABLE_PATH)  
        results_adr.append(pdr)
        csv_path = os.path.join("results", f"topologia_final_adr_{count}n.csv")
        png_path = os.path.join("results", f"topologia_sf_adr_{count}n.png")
        plot_sf_distribution(csv_path, png_path)

    print("\n\n--- Todas las simulaciones completadas. Generando gráfico final... ---")
    plt.figure(figsize=(10, 6))
    plt.plot(NODE_TEST_COUNTS, results_qlearning, label="Q-Learning (Network-Aware)", marker='o', markersize=8, linewidth=2)
    plt.plot(NODE_TEST_COUNTS, results_adr, label="ADR Nativo (Link-Based)", marker='x', markersize=8, linestyle='--', linewidth=2)
    plt.title(f"Comparativa de Escalabilidad (PDR vs Densidad)\n({len(GW_POSITIONS)} GWs, {N_CHANNELS} Canales)")
    plt.xlabel("Número de Nodos en la Red (Densidad)")
    plt.ylabel("PDR Global (%)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    final_graph_file = os.path.join("results", "comparativa_escalabilidad.png")
    plt.savefig(final_graph_file)
    plt.close()
    print(f"Gráfico de escalabilidad guardado en: {final_graph_file}")
    print("\nResultados (Q-Learning):", results_qlearning)
    print("Resultados (ADR):", results_adr)