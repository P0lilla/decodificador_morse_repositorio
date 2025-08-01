# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 16:17:16 2025

@author: pacoe
"""
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

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

###### CARGA DE ARCHIVO Y ANÁLISIS PRELIMINAR ######

def carga_audio():
    ruta_archivo = input('Introduce la ruta del archivo: ')
    
    # Si la ruta comienza con " o ' los eliminamos.
    if ruta_archivo[0] in ("'", '"'):
        ruta_archivo = ruta_archivo[1:-1]
    
    sample_rate, data = wavfile.read(ruta_archivo)
    
    # Conocemos la duración de la pista gracias a que los Hz indican cuantos 
    # valores aparecen por segundo.
    duracion = round(1/(sample_rate/len(data)), 4)
    
    print(f"Tasa de muestreo: {sample_rate} Hz")
    print(f"Forma del array de datos: {data.shape}")
    print(f"Tipo de datos: {data.dtype}")
    print(f'Duración: {duracion}s')
    print()
    return data, duracion, ruta_archivo

###### NORMALIZAMOS LOS DATOS A FLOAT32 ######

def normalizar_codificacion(data):
    # Las formas más comunes de codificar valores son usando:
        # uint8: 8 bits sin signo, rango de 0 a 255 (silencio = 128)
        # int16: 16 bits con signo, rango de -32768 a 32767.
        # int32: 32 bits con signo, rango -2.147.483.648 a +2.147.483.647
        # float32: punto flotante entre -1.0 y 1.0.
        
    # Si los datos están en estéreo, seleccionamos solo la pista L.
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
    else:
        print(f'INVALID FORMAT: {tipo}')
    return data

###### REPRESENTACIÓN GRÁFICA DE LA ONDA SONORA ######

def representacion_grafica(duracion, data, ruta_archivo):
    # Comenzamos la represntación gráfica definiendo un vector tiempo que va 
    # desde  0 hasta la duración de la pista.

    tiempo = np.linspace(0, duracion, num=len(data))

    # Graficamos con mathplotlib.
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, data)
    plt.title(f"Onda sonora: {ruta_archivo.split('\\')[-1]}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud normalizada")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

###### TRANSFORMACIÓN DE ONDA A CÓDIGO MORSE ######

# El primer paso es transformar el conjunto de puntos de amplitud que 
# conforman la onda en una lista de tuplas que indican los pulsos que la 
# componen. Un pulso es (inicio_sonido, fin_sonido). Como la amplitud de la 
# onda fluctúa entre valores positivo y negativos, los tonos del código morse
# se descomponen en multitud de pequeños pulsos.

def onda_a_pulsos(data):
    # Introducimos un indicador de final de pista (un valor elevado al final
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

    # Extraemos listas que contienen los índices donde comienza y finaliza
    # el sonido np.where devuelve una tupla de la que solo nos interesa
    # el primer elemento.
    inicios = np.where(cambios == 1)[0]
    finales = np.where(cambios == -1)[0]

    if inicios[0] < finales[0]:
        # Creamos una lista de tuplas de pulsos (inicio, fin)
        pulsos = list(zip(inicios, finales))
        for pulso in pulsos:
            if pulso[0] > pulso[1]:
                print('ERROR EN EL ORDEN DE TUPLA PULSO')
    return pulsos
    
# El siguiente paso es agrupar los pulsos que estén muy juntos entre sí
# para formar los tonos (cortos y largos) que componen el mensaje en morse.
def pulsos_a_tonos(pulsos):
    # Recorremos la lista de pulsos. Si la distancia entre el final de uno 
    # y el comienzo de otro es pequeña, los fusionamos.
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
    return pulsos_limpios

# El siguiente paso es transformar la lista de tonos, que solo indica 
# el comienzo y el final de cada tono, en una lista de tuplas que calcule 
# también los silencios e indique si se trata de un tono (corto o largo)
# o de una pausa (corta entre tonos de una letra, media entre letras y larga
# entre palabras). 
def clasificacion_tonos_y_silencios(tonos_morse):
    # Creamos una lista con tuplas (longitud, tipo) que intercala los tonos con
    # los silencios indicando la duración de cada uno para diferenciar entre 
    # tonos cortos y largos y pausas intra-letra, inter-letra y inter-palabra.
    tonos_y_pausas = []

    for index in range(len(tonos_morse)):
        tono = tonos_morse[index]
        duracion_tono = tono[1] - tono[0]
        tonos_y_pausas.append((duracion_tono, 'tono'))

        # Añadimos pausa si existe un siguiente tono
        if index < len(tonos_morse) - 1:
            tono_siguiente = tonos_morse[index + 1]
            duracion_pausa = tono_siguiente[0] - tono[1]
            tonos_y_pausas.append((duracion_pausa, 'pausa'))

    # Calculamos el valor máximo y el mínimo entre los tonos y las pausas.
    max_tono = max([x[0] for x in tonos_y_pausas if x[1] == "tono"])
    min_tono = min([x[0] for x in tonos_y_pausas if x[1] == "tono"])

    max_pausa = max([x[0] for x in tonos_y_pausas if x[1] == "pausa"])
    min_pausa = min([x[0] for x in tonos_y_pausas if x[1] == "pausa"])



    # Clasificación de tonos y pausas en corta, media o larga atendiendo a 
    # su longitud relativa a los máximos y los mínimos.
    tonos_y_silencios_clasificados = []
    for valor, etiqueta in tonos_y_pausas:
        if etiqueta == 'tono':
            if abs(valor - min_tono) < (min_tono/10):
                clasificacion = 'Tono corto'
            else:
                clasificacion = 'Tono largo'
            tonos_y_silencios_clasificados.append([valor, clasificacion])
        
        else:
            if abs(valor - min_pausa) < (min_pausa/10):
                clasificacion = 'Pausa corta'
            elif abs(valor - max_pausa) < (max_pausa/10):
                clasificacion = 'Pausa larga'
            else:
                clasificacion = 'Pausa media'
            tonos_y_silencios_clasificados.append([valor, clasificacion])

    # Si solo tenemos una palabra, las pausas inter-letra se interpretan como
    # inter- palabra. Lo corregimos tal que: si no hay "Pausas medias", las
    # pausas largas se sustituyen por pausas medias.
    hay_medios = any(x[1] == "Pausa media" for x in tonos_y_silencios_clasificados)

    if not hay_medios:
        print('JUST ONE WORD DETECTED.')
        tonos_y_silencios_clasificados = [[x[0], "Pausa media" if x[1] == "Pausa larga" else x[1]] for x in tonos_y_silencios_clasificados]
    
    return tonos_y_silencios_clasificados

# El siguiente paso es transformar la lista de tonos y silencios en su 
# equivalente en morse escrito.
def a_morse_escrito(tonos_y_silencios_clasificados):
    letra = ''
    palabra = ''
    mensaje = ''
    for n in tonos_y_silencios_clasificados:
        tipo = n[1]
        
        if tipo == 'Tono corto':
            letra += '.'
        elif tipo == 'Tono largo':
            letra += '-'
        elif tipo == 'Pausa media':
            if letra:
                letra += ' '
                palabra += letra
                letra = ''
        elif tipo == 'Pausa larga':
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
        
    print(f'El audio corresponde al mensaje en morse: {mensaje}')
    return mensaje
    
# El último paso es transformar el mensaje de morse escrito a su equivalente
# latino.
def morse_a_latino(mensaje):
    # Transformamos el mensaje en una lista de palabras separadas por espacios.
    mensaje = mensaje.split("  / ") 
    traduccion = ''
    for palabra in mensaje:
        letras = palabra.split(' ')
        for letra in letras:
            if letra in morse_to_char:
                traduccion += morse_to_char[letra]
            else:
                traduccion += '#'
        traduccion += ' '
        
    print(f'El mensaje en morse se traduce a: {traduccion}')
    return(traduccion)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
data, duracion, ruta_archivo = carga_audio()  
data = normalizar_codificacion(data)
representacion_grafica(duracion, data, ruta_archivo)
pulsos = onda_a_pulsos(data)
tonos_morse = pulsos_a_tonos(pulsos)
tonos_y_silencios_clasificados = clasificacion_tonos_y_silencios(tonos_morse)
mensaje = a_morse_escrito(tonos_y_silencios_clasificados)
traduccion = morse_a_latino(mensaje)
