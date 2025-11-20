import math
from parameters import *

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Funciones Físicas y del Entorno
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def sensitivity_dbm(sf):                                                        # Define la sensibilidad mínima de recepción del gateway para cada Spreading Factor (SF).
    return {7: -123, 8: -126, 9: -129, 10: -132, 11: -134.5, 12: -137}[sf]      # Diccionario que actúa como una tabla de consulta. Si le pides

def log_distance_pl_db(distance_m, freq_mhz, n):                                # Define la función de pérdida de la señal. (Log distance path loss)
    d = max(distance_m, 1.0)                                                    # Asegura que la distancia nunca sea cero para evitar error matemático de -inf. (min 1m)
    freq_hz = freq_mhz * 1e6                                                    # Convierte la frecuencia de MHz a hZ
    fspl_1m = 20.0 * math.log10(4.0 * math.pi * freq_hz / C)                    # Pédida en espacio libre a d_0 = 1m: FSPL = PL(d_0) = 20log(d*f*4*pi/c)
    return fspl_1m + 10.0 * n * math.log10(d / 1.0)                             # Fórmula de Log-Distance Path Loss: PL = PL(d_0) + 10*n*log(d/d_0)

def okumura_hata_pl_db(distance_m, freq_mhz):
    d_km = max(0.01, distance_m / 1000.0)                                       # Distancia en km
    hb = 15.0                                                                    # Altura de la antena base (Gateway) en metros
    hm = 1.5                                                                    # Altura de la antena móvil (Nodo) en metros
    ahm = (1.1 * math.log10(freq_mhz) - 0.7) * hm - (1.56 * math.log10(freq_mhz) - 0.8)
    Lp_urban = (69.55 + 26.16 * math.log10(freq_mhz) 
                - 13.82 * math.log10(hb) - ahm 
                + (44.9 - 6.55 * math.log10(hb)) * math.log10(d_km))
    Lp_rural = Lp_urban - (4.78 * (math.log10(freq_mhz)**2)) + (18.33 * math.log10(freq_mhz)) - 40.94
    return Lp_rural

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
