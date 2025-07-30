# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:30:19 2025

@author: pacoe

Los Hz indican cuantas muestras hay por segundo.
data.shape indica cuántas muestras hay y si sólo hay un canal (mono) o varios.


"""
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# wavfile devuelve una tupla con los Hz de la pista y un array Numpy con las
# muestras de audio.
# Los Hz indican cuantos valores hay por segundo.
# Cada valor del array indica la amplitud de la onda en ese instante de tiempo.
ruta_archivo = r'D:\OneDrive\Universidad\Máster Bioinformática\Proyectos personales\8. Decodificador Morse\morse-SOS-20wpm-600hz.wav'
sample_rate, data = wavfile.read(ruta_archivo)
# Análisis inicial:
duracion = round(1/(sample_rate/len(data)), 4)

print(f"Tasa de muestreo: {sample_rate} Hz")
print(f"Forma del array de datos: {data.shape}")
print(f"Tipo de datos: {data.dtype}")
print(f'Duración: {duracion}s')

# Definimos una función para normalizar la codificación a float32.
def normalizar_codificacion(data):
    # Las formas más comunes de codificar valores son usando:
        # uint8: 8 bits sin signo, rango de 0 a 255 (silencio = 128)
        # int16: 16 bits con signo, rango de -32768 a 32767. Muy utilizado.
        # int32: 32 bits con signo, rango -2.147.483.648 a +2.147.483.647
        # float32: punto flotante entre -1.0 y 1.0.
    # Comenzamos fusionando las pistas stereo en mono.
    if len(data.shape) == 2 and data.shape[1] == 2:
        print('STEREO DETECTED. FUSING TO MONO.')
        data = data.mean(axis=1)
    else:
        print('MONO DETECTED.')

    # Para representaciones gráficas, usamos la forma float32.
    tipo = data.dtype
    if tipo == np.uint8:
        print('UINT8 DETECTED')
        data = data.astype(np.float32)
        data = (data - 128) / 128
    elif tipo == np.int16:
        print('INT16 DETECTED')
        data = data.astype(np.float32) / 32768.0
    elif tipo == np.int32:
        print('INT32 DETECTED')
        data = data.astype(np.float32) / 2147483648.0
    elif tipo == np.float64:
        print('FLOAT64 DETECTED')
        data = data.astype(np.float32)
    elif tipo == np.float32:
        print('FLOAT32 DETECTED')
        print(f'Rango: {np.min(data)} a {np.max(data)}')
    else:
        print(f'INVALID FORMAT: {tipo}')
    return data

# Normalizamos los datos aplicando la funciñon normalizar_codificacion.
data = normalizar_codificacion(data)

# Comenzamos la represntación gráfica definiendo un vector tiempo que va desde
# 0 hasta la duración de la pista.

tiempo = np.linspace(0, duracion, num=len(data))

# Graficamos.
plt.figure(figsize=(10, 4))
plt.plot(tiempo, data)
plt.title(f"Onda sonora: {ruta_archivo.split('\\')[-1]}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud normalizada")
plt.grid(True)
plt.tight_layout()
plt.show()

# ANÁLISIS DE SEGMENTOS.

# Calculamos la amplitud absoluta y establecemos un umbral de sonido.
amplitud = np.abs(data)
umbral = 0.2

# Comparamos la amplitud absoluta al umbral para generar una lista booleana.
actividad = amplitud > umbral

# Cambiamos los booleanos a 0 - 1 con astype(int) y calculamos
# la diferencia b - a entre los valores consecutivos del vector para
# identificar cuando comienza (1) y cuando acaba (-1) el sonido.
cambios = np.diff(actividad.astype(int))

# Extraemos listas que contienen los índices donde comienza y finaliza el sonido.
# np.where devuelve una tupla de la que solo nos interesa el primer elemento.
inicios = np.where(cambios == 1)[0]
finales = np.where(cambios == -1)[0]

if inicios[0] < finales[0]:
    print('Primera posición = inicio sonido')
    print(f'Hay {len(inicios)} inicios y {len(finales)} finales')
    # Creamos una lista de tuplas de pulsos (inicio, fin)
    pulsos = list(zip(inicios, finales))
    for pulso in pulsos:
        if pulso[0] > pulso[1]:
            print('ERROR EN EL ORDEN DE TUPLA PULSO')
    
# Recorremos la lista de pulsos. Si la distancia entre el final de uno y el 
# comienzo de otro es pequeña, los fusionamos.
swich = True

for i in range(len(pulsos)-1):
    pulso_actual = pulsos[i]
    pulso_siguiente = pulsos[i+1]
    
    if (pulso_siguiente[0] - pulso_actual[1]) > 10:
        print(f'posible salto de pulso entre {pulso_actual} y {pulso_siguiente}')