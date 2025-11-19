import numpy as np

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
PATH_LOSS_EXPONENT = 2.8                         # Exponente de atenuación del modelo de ptopagación
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
    (0, 0),                                       # Cantidad y ubicación de GW en la simulación
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

CO_CHANNEL_REJECTION_DB = {
    # SF Interferente:
    # SFd/SFi  7     8     9     10    11    12
    7:      { 7: 6,  8:16,  9:18, 10:19, 11:19, 12:20 },
    8:      { 7:24,  8: 6,  9:20, 10:22, 11:22, 12:22 },
    9:      { 7:27,  8:27,  9: 6, 10:23, 11:25, 12:25 },
    10:     { 7:30,  8:30,  9:30, 10: 6, 11:26, 12:28 },
    11:     { 7:33,  8:33,  9:33, 10:33, 11: 6, 12:29 },
    12:     { 7:36,  8:36,  9:36, 10:36, 11:36, 12: 6 }
}