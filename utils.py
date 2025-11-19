import os
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt 
import pandas as pd
from parameters import *
from entities import *

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


def plot_sf_distribution(csv_filename, output_png):
    """
    Crea un gráfico de dispersión de la topología de nodos, 
    coloreado por SF.
    """
    if not os.path.exists(csv_filename):
        print(f"ADVERTENCIA: No se encontró el archivo de topología {csv_filename}. No se generará el gráfico.")
        return
        
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error al leer {csv_filename}: {e}")
        return
    
    plt.figure(figsize=(12, 10))
    
    sf_values = sorted(df['final_sf'].dropna().unique())
    colors = plt.cm.jet(np.linspace(0, 1, len(sf_values)))
    sf_color_map = {sf: color for sf, color in zip(sf_values, colors)}
    sf_color_map[np.nan] = 'gray'

    for sf, color in sf_color_map.items():
        if pd.isna(sf):
            subset = df[df['final_sf'].isnull()]
            label = "SF No Asignado"
        else:
            subset = df[df['final_sf'] == sf]
            label = f"SF {int(sf)}"
        
        if not subset.empty:
            plt.scatter(subset['x'], subset['y'], c=[color], label=label, alpha=0.7, s=20)
    
   
    gw_x = [pos[0] for pos in GW_POSITIONS]
    gw_y = [pos[1] for pos in GW_POSITIONS]
    plt.scatter(gw_x, gw_y, c='red', marker='^', s=200, label="Gateway", edgecolors='black', zorder=10)
    
  
    plt.title(f"Distribución de Nodos y Spreading Factor (SF)\n({os.path.basename(output_png)})", fontsize=16)
    plt.xlabel("Coordenada X (m)", fontsize=12)
    plt.ylabel("Coordenada Y (m)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.savefig(output_png)
    plt.close()
    print(f"Gráfico de topología guardado en: {output_png}")