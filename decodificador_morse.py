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
from collections import Counter

# wavfile devuelve una tupla con los Hz de la pista y un array Numpy con las
# muestras de audio.
# Los Hz indican cuantos valores hay por segundo.
# Cada valor del array indica la amplitud de la onda en ese instante de tiempo.
ruta_archivo = r'D:\OneDrive\Universidad\Máster Bioinformática\Proyectos personales\8. Decodificador Morse\morse.wav'
sample_rate, data = wavfile.read(ruta_archivo)
# Análisis inicial:
duracion = round(1/(sample_rate/len(data)), 4)

print(f"Tasa de muestreo: {sample_rate} Hz")
print(f"Forma del array de datos: {data.shape}")
print(f"Tipo de datos: {data.dtype}")
print(f'Duración: {duracion}s')
print()

# Definimos una función para normalizar la codificación a float32.
def normalizar_codificacion(data):
    # Las formas más comunes de codificar valores son usando:
        # uint8: 8 bits sin signo, rango de 0 a 255 (silencio = 128)
        # int16: 16 bits con signo, rango de -32768 a 32767. Muy utilizado.
        # int32: 32 bits con signo, rango -2.147.483.648 a +2.147.483.647
        # float32: punto flotante entre -1.0 y 1.0.
    # Comenzamos fusionando las pistas stereo en mono.
    if len(data.shape) == 2 and data.shape[1] == 2:
        print('STEREO DETECTED. L CHANNEL SELECTED.')
        data = data[:, 0]
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

# Introducimos un indicador de inicio y final de pista (un valor elevado al final
# segudo de 0)
indicador = np.array([0.999, 0])
data = np.concatenate((data, indicador))
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
contador = 0
silencio_intratono = 30
nuevo_pulso = [None, None]
pulsos_limpios = []
for i in range(len(pulsos)-1):
    inicio_pulso_actual = pulsos[i][0]
    inicio_pulso_siguiente = pulsos[i + 1][0]
    
    fin_pulso_actual = pulsos[i][1]
    fin_pulso_siguiente = pulsos[i + 1][1]
    
    # Si la diferencia entre un pulso y el siguient es muy pequeña
    if (inicio_pulso_siguiente - fin_pulso_actual) < silencio_intratono:
        # y si no hay un nuevo pulso en construcción
        if nuevo_pulso[0] is None:
            # El nuevo pulso es el inicio del primero y el fin del segundo.
            nuevo_pulso[0] = inicio_pulso_actual
            nuevo_pulso[1] = fin_pulso_siguiente
        # Pero si hay un nuevo pulso en construcción:    
        else:
            # Solo actualizamos el final del pulso.
            nuevo_pulso[1] = fin_pulso_siguiente
    # Si la diferencia entre un pulso y el siguiente es grande
    elif (inicio_pulso_siguiente - fin_pulso_actual) > silencio_intratono:
        # Si hay un pulso en construcción:
        if nuevo_pulso[0] is not None:
            # Introducimos el nuevo pulso en la lista buena y lo reseteamos.
            pulsos_limpios.append(nuevo_pulso)
            contador += 1
            nuevo_pulso = [None, None]
# Si se ha quedado un último pulso pendiente, lo metemos también.
if nuevo_pulso[0] is not None:
    pulsos_limpios.append(nuevo_pulso)
    contador += 1

print(f'Detectados {contador} pulsos')            
print()

# Creamos una lista con tuplas (longitud, tipo) que intercala los sonidos con
# las pausas indicando la duración de cada uno para diferenciar entre 
# sonidos cortos y largos y pausas intra-letra, inter-letra y inter-palabra.
codigo = []

for index in range(len(pulsos_limpios)):
    pulso = pulsos_limpios[index]
    duracion_pulso = pulso[1] - pulso[0]
    codigo.append((duracion_pulso, 'pulso'))

    # Añadimos pausa si existe un siguiente pulso
    if index < len(pulsos_limpios) - 1:
        pulso_siguiente = pulsos_limpios[index + 1]
        duracion_pausa = pulso_siguiente[0] - pulso[1]
        codigo.append((duracion_pausa, 'silencio'))


max_pulso = max([x[0] for x in codigo if x[1] == "pulso"])
min_pulso = min([x[0] for x in codigo if x[1] == "pulso"])

max_silencio = max([x[0] for x in codigo if x[1] == "silencio"])
min_silencio = min([x[0] for x in codigo if x[1] == "silencio"])



# Clasificación en corta, media o larga
clasificadas = []
for valor, etiqueta in codigo:
    if etiqueta == 'pulso':
        if abs(valor - min_pulso) < (min_pulso/10):
            clasificacion = 'Pulso corto'
        else:
            clasificacion = 'Pulso largo'
        clasificadas.append([valor, clasificacion])
    
    else:
        if abs(valor - min_silencio) < (min_silencio/10):
            clasificacion = 'Silencio corto'
        elif abs(valor - max_silencio) < (max_silencio/10):
            clasificacion = 'Silencio largo'
        else:
            clasificacion = 'Silencio medio'
        clasificadas.append([valor, clasificacion])

# Si solo tenemos una palabra, los espacios inter-letra se interpretan como
# inter- palabra. Lo corregimos tal que: si no hay "Silencios medios", los
# silencios largos se sustituyen por silencios medios.
hay_medios = any(x[1] == "Silencio medio" for x in clasificadas)

if not hay_medios:
    print('JUST ONE WORD DETECTED.')
    clasificadas = [[x[0], "Silencio medio" if x[1] == "Silencio largo" else x[1]] for x in clasificadas]

        
print(Counter(codigo))    
print()

# Transformación de pulsos a morse escrito.
letra = ''
palabra = ''
mensaje = ''
for n in clasificadas:
    tipo = n[1]
    
    if tipo == 'Pulso corto':
        letra += '.'
    elif tipo == 'Pulso largo':
        letra += '-'
    elif tipo == 'Silencio medio':
        if letra:
            letra += ' '
            palabra += letra
            letra = ''
    elif tipo == 'Silencio largo':
        if letra:
            letra += ' '
            palabra += letra
            letra = ''
        if palabra:
            mensaje += palabra + ' / '
            palabra = ''
if letra:
    palabra += letra
if palabra:
    mensaje += palabra
    
# Transformamos el mensaje en una lista de palabras separadas por espacios.
mensaje = mensaje.split("  / ")    

# Diccionario morse clave símbolos morse, valor letra latina.
morse_to_char = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', 
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', 
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O', 
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T', 
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y', 
    '--..': 'Z', '..-..': 'É', '-----': '0', '.----': '1', 
    '..---': '2', '...--': '3', '....-': '4', '.....': '5', 
    '-....': '6', '--...': '7', '---..': '8', '----.': '9', 
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '-.-.--': '!', 
    '-....-': '-', '-..-.': '/', '.--.-.': '@', '-.--.': '(', 
    '-.--.-': ')', '---...': ':'
}

traduccion = ''
for palabra in mensaje:
    letras = palabra.split(' ')
    for letra in letras:
        if letra in morse_to_char:
            traduccion += morse_to_char[letra]
        else:
            traduccion += '#'
    traduccion += ' '
    
print(traduccion)