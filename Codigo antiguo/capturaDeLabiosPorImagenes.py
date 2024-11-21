import time

import cv2
import dlib
import math
import re
import json
from collections import Counter
import random
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from jaxlib.xla_extension.ops import Reshape
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Masking, Bidirectional
from keras.src.utils import to_categorical
from num2words import num2words

from pytube import YouTube
from unidecode import unidecode
from youtube_transcript_api import YouTubeTranscriptApi
import os

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Concatenate, Dropout, \
    TimeDistributed, BatchNormalization
from tensorflow.keras.models import Model

##
#imports tensorflow
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import glob
import collections
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import  locale

from keras.src.saving import register_keras_serializable
locale.setlocale(locale.LC_ALL,'es_ES')

from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers

#Wilcoxon Test
import warnings
warnings.filterwarnings('ignore')

import cv2

##################################################
#########       Variables globales       #########
##################################################

nombreModelo = "3_variables_25_epocas_tiempo_real"

resultadosROC = []

# Cargar el detector de caras y el predictor de puntos clave de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

key_points_interesantes = [48, 49, 50, 52, 53, 54 , 55, 56, 58, 59, 62, 66]

# Lista de rutas de imágenes a procesar
rutas_imagenes = ["../Materiales/Imagenes/prueba.png","../Materiales/Imagenes/prueba.jpg","../Materiales/Imagenes/girada.jpg","../Materiales/Imagenes/Vocales/nada.jpg","../Materiales/Imagenes/Vocales/a.jpg","../Materiales/Imagenes/Vocales/e.jpg","../Materiales/Imagenes/Vocales/i.jpg","../Materiales/Imagenes/Vocales/o.jpg","../Materiales/Imagenes/Vocales/u.jpg"]  # Agrega las rutas de tus imágenes aquí

#Maximo de frames 219 de consuelo
#Primer frame frame_50122.jpg y ultimo frame_50339.jpg
##################################################
#########         Cargar Datos           #########
##################################################

def leer_srt(archivo):
    # Leer el contenido del archivo .srt
    with open(archivo, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Inicializar un array para almacenar los resultados
    word_number_pairs = []

    # Recorrer las líneas y extraer la información
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            number = int(lines[i].strip())
            i += 2  # Saltar la línea de tiempo
            word = lines[i].strip()
            word = tratar_palabra(word)

            word_number_pairs.append((word, number))
        i += 1

    # Mostrar el resultado
    for pair in word_number_pairs:
        print(pair)

    return word_number_pairs

def recogiendo_frames_de_cada_palabra(array_de_tuplas,directorio):
    # Crear un diccionario para almacenar los resultados
    resultados = []

    # Recorrer cada tupla
    for i in range(len(array_de_tuplas)):
        numero = array_de_tuplas[i][1]
        array_frames = []
        # Crear un patrón de búsqueda usando regex para el formato frame_{numero}_algo.png
        patron = re.compile(rf"^frame_{numero}_\d+\.png$")

        # Recorrer los archivos en el directorio
        for archivo in os.listdir(directorio):
            if patron.match(archivo):
                array_frames.append(archivo)

        array_frames_sorted = sorted(array_frames, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        resultados.append((array_de_tuplas[i][0],array_frames_sorted))

    # Imprimir los resultados
    for letra, archivos in resultados:
        print(f"{letra}: {archivos}")

    return resultados

def tratar_palabra(palabra):
    if palabra.isdigit():
        palabra = num2words(palabra, lang='es')
    palabra = unidecode(palabra).lower()
    palabra_final = ''.join(c for c in palabra if c.isalpha() or c.isspace())
    return palabra_final

def verificar_datos(tuplas_letra_frames,ruta_fija):
    contador = 0
    tuplas_depuradas = tuplas_letra_frames.copy()
    for tupla in tuplas_letra_frames:
        contador += 1
        print(f"{contador} array es de {len(tuplas_depuradas)}")
        for frame in tupla[1]:
            ruta = f'{ruta_fija}/{frame}'
            imagen = cv2.imread(ruta)
            lips_points = son_labios_detectados(imagen)
            if len(lips_points) == 0 or None in lips_points:
                tuplas_depuradas.remove(tupla)
                break

    return tuplas_depuradas

def extraer_lips_points_datos(tuplas_letra_frames,ruta_fija):
    contador = 0
    tuplas_depuradas = [list(tupla) for tupla in tuplas_letra_frames]  # Convertir tuplas internas a listas
    for i in range(len(tuplas_letra_frames)):
        frames = tuplas_letra_frames[i][1]
        contador += 1
        print(f"{contador} de {len(tuplas_depuradas)}")
        lips_points_array = []
        for l in range(len(frames)):
            ruta = f'{ruta_fija}/{frames[l]}'
            imagen = cv2.imread(ruta)
            lips_points = son_labios_detectados(imagen)
            if len(lips_points) == 0 or None in lips_points:
                tuplas_depuradas.remove(list(tuplas_letra_frames[i]))
                break
            else:
                lips_points_array.append(normalize_keypoints(lips_points))

        tuplas_depuradas[i][1] = lips_points_array

    return numpy_array_to_list(tuplas_depuradas)


def son_labios_detectados(imagen):
    imagen_alterada = imagen.copy()
    lips_points = []

    # Detectar caras
    faces = detector(imagen_alterada)

    # Para cada cara detectada, predecir los puntos clave
    for face in faces:

        landmarks = predictor(imagen_alterada, face)

        lips_points = []

        for n in key_points_interesantes:  # Los índices de los puntos de los labios en el modelo de 68 puntos
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lips_points.append((x, y))
        break

    return lips_points

def numpy_array_to_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, list):
        return [numpy_array_to_list(item) for item in arr]
    else:
        return arr

def json_list_to_numpy(arr):
    if isinstance(arr, list):
        return [json_list_to_numpy(item) for item in arr]
    elif isinstance(arr, dict):
        try:
            return np.array(arr['data'], dtype=np.float64).reshape(arr['shape'])
        except KeyError:
            return arr
    else:
        return arr


def guardar_array_en_json(array, nombre_json, partes=10):
    if os.path.exists(nombre_json):
        os.remove(nombre_json)

    array = numpy_array_to_list(array)
    longitud = len(array)
    chunk_size = longitud // partes
    archivos_generados = []

    for i in range(partes):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i < partes - 1 else longitud
        chunk = array[start_index:end_index]
        json_data = json.dumps(chunk, indent=2)
        archivo_parcial = f"{nombre_json.rsplit('.', 1)[0]}_parte_{i + 1}.json"

        with open(archivo_parcial, 'w') as f:
            f.write(json_data)

        archivos_generados.append(archivo_parcial)

    return archivos_generados


def cargar_array_desde_json(archivos_json):
    array_completo = []
    for archivo in archivos_json:
        with open(archivo, 'r') as f:
            chunk = json.load(f)
            array_completo.extend(chunk)
    return array_completo

##################################################
#########         Extraer Datos          #########
##################################################
def extraer_lips_points(imagen):
    imagen_alterada = imagen.copy()
    lips_points = []
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen_alterada, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faces = detector(gray)

    # Para cada cara detectada, predecir los puntos clave
    for face in faces:
        # Dibujar un recuadro azul alrededor de la cara
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(imagen_alterada, (x, y), (x + w, y + h), (255, 0, 0), 2)  # (255, 0, 0) es azul en BGR

        landmarks = predictor(gray, face)

        # Extraer los puntos clave de los labios
        lips_points = []

        for n in key_points_interesantes:  # Los índices de los puntos de los labios en el modelo de 68 puntos
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lips_points.append((x, y))
            cv2.circle(imagen_alterada, (x, y), 1, (0, 255, 0), -1)
        break

    # Mostrar la imagen con los puntos clave dibujados
    cv2.imshow("Lips Keypoints", imagen_alterada)
    cv2.waitKey(0)  # Esperar hasta que se presione una tecla
    cv2.destroyAllWindows()

    return lips_points

##################################################
#########         Normalizacion          #########
##################################################
def redimensionar_imagen_gris(dimensiones_imagen,imagen):
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(imagen_gris, dimensiones_imagen)
    cv2.imshow('Imagen en escala de grises', resized_img)
    return resized_img

def angulo_boca(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle = math.atan2(delta_y, delta_x)
    return math.degrees(angle)

def rotar_imagen_angulo_0(imagen,angulo,point1,point2):
    center = ((point1[0]+point2[0])//2, (point1[1]+point2[1])//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angulo, 1.0)
    imagen_rotada = cv2.warpAffine(imagen, rotation_matrix, (imagen.shape[1], imagen.shape[0]))
    cv2.imwrite('imagen_prueba_rotada.jpg', imagen_rotada)
    return imagen_rotada

def normalize_keypoints(all_lips_points):
    all_lips_points = np.array(all_lips_points)
    min_vals = all_lips_points.min(axis=0)
    max_vals = all_lips_points.max(axis=0)
    normalized_points = (all_lips_points - min_vals) / (max_vals - min_vals)
    return normalized_points

def recortar_imagen_labios(lips_points, imagen, padding_vertical=8, padding_lateral=8):
    # Convertir los puntos de los labios a un array de numpy
    lips_points = np.array(lips_points)

    # Obtener los valores mínimos y máximos en los ejes x e y
    min_vals = lips_points.min(axis=0)
    max_vals = lips_points.max(axis=0)

    # Calcular los nuevos límites con padding
    top_left_x = max(min_vals[0] - padding_lateral, 0)
    top_left_y = max(min_vals[1] - padding_vertical, 0)
    bottom_right_x = min(max_vals[0] + padding_lateral, imagen.shape[1])
    bottom_right_y = min(max_vals[1] + padding_vertical, imagen.shape[0])

    # Recortar la imagen
    imagen_recortada = imagen[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return imagen_recortada

def resize_image(image, target_size=(128, 128)):
    # Redimensionar la imagen manteniendo la relación de aspecto
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Crear una imagen en blanco con las dimensiones finales
    new_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    # Centramos la imagen redimensionada en la imagen nueva
    y_offset = (target_size[0] - resized.shape[0]) // 2
    x_offset = (target_size[1] - resized.shape[1]) // 2
    new_image[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

    cv2.imshow("Labios", new_image)

    return new_image


def resize_and_pad(image, output_size=(32, 32), padding_color=(0,0,0)):
    """
    Redimensiona una imagen manteniendo su relación de aspecto y añade padding para ajustarla
    a un tamaño específico, luego normaliza la imagen.

    Parámetros:
    image (np.ndarray): Imagen original.
    output_size (tuple): Tamaño de salida deseado (ancho, alto).
    padding_color (int): Color de padding en escala de grises (0-255).

    Devuelve:
    np.ndarray: Imagen preprocesada con forma (output_size[1], output_size[0], 1) y valores normalizados entre 0 y 1.
    """
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Obtener las dimensiones originales de la imagen
    h, w = imagen_gris.shape[:2]

    # Calcular la relación de aspecto
    aspect_ratio = w / h

    # Determinar la nueva dimensión
    if aspect_ratio > 1:
        # La imagen es más ancha que alta
        new_w = output_size[0]
        new_h = int(output_size[0] / aspect_ratio)
    else:
        # La imagen es más alta que ancha
        new_h = output_size[1]
        new_w = int(output_size[1] * aspect_ratio)

    # Redimensionar la imagen manteniendo la relación de aspecto
    resized_image = cv2.resize(imagen_gris, (new_w, new_h))

    # Calcular el padding necesario
    top = (output_size[1] - new_h) // 2
    bottom = output_size[1] - new_h - top
    left = (output_size[0] - new_w) // 2
    right = output_size[0] - new_w - left

    # Añadir el padding
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                      value=padding_color)


    return padded_image

def normalizar_imagenes(imagen):
    # Normalizar los valores de los píxeles al rango [0, 1]
    imagen_normalizada = imagen / 255.0

    # Añadir una dimensión para el canal de color
    imagen_normalizada_con_canal = np.expand_dims(imagen_normalizada, axis=-1)

    return imagen_normalizada_con_canal


def desplazar_imagen(image, shift_x, padding_color=(0, 0, 0)):
    """
    Desplaza la imagen horizontalmente a la izquierda o derecha.

    :param image: La imagen a desplazar.
    :param shift_x: Cantidad de píxeles a desplazar. Un valor positivo desplaza a la derecha, un valor negativo desplaza a la izquierda.
    :param padding_color: Color de relleno para los espacios vacíos. Por defecto es negro.
    :return: La imagen desplazada.
    """
    # Obtener las dimensiones de la imagen
    h, w = image.shape[:2]

    # Crear una matriz de transformación para el desplazamiento
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])

    # Aplicar la transformación para desplazar la imagen
    shifted_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=padding_color)

    return shifted_image

def guardar_imagen_cv2(imagen, ruta_destino):
    """
    Guarda una imagen en la ruta especificada utilizando OpenCV.

    Args:
    imagen (numpy.ndarray): La imagen a guardar.
    ruta_destino (str): La ruta donde se guardará la imagen, incluyendo el nombre del archivo y la extensión.
    """
    cv2.imwrite(ruta_destino, imagen)


def contar_palabras(lista_palabras):
    if isinstance(lista_palabras, np.ndarray):
        lista_palabras = lista_palabras.tolist()

    # Usamos Counter para contar la frecuencia de cada palabra
    frecuencia_palabras = Counter(lista_palabras)

    # Ordenamos las palabras por frecuencia en orden descendente
    palabras_ordenadas = sorted(frecuencia_palabras.items(), key=lambda x: x[1], reverse=True)

    # Creamos el resultado con cada palabra en una línea
    resultado = '\n'.join(f'{palabra}: {frecuencia}' for palabra, frecuencia in palabras_ordenadas)

    return resultado

def mostrar_imagen(imagen):
    cv2.imshow("Mostrando imagen", imagen)
    cv2.waitKey(0)  # Esperar hasta que se presione una tecla
    cv2.destroyAllWindows()

def contar_palabras_diferentes(array):
    # Crear un conjunto para almacenar las palabras únicas
    palabras_unicas = set()

    # Iterar sobre cada palabra en el array
    for palabra in array:
        # Agregar la palabra al conjunto (los duplicados se eliminan automáticamente)
        palabras_unicas.add(palabra)

    # Retornar la cantidad de palabras únicas
    return len(palabras_unicas)

def transformar_one_hot(array):
    # Obtener vocabulario único
    vocabulario = np.unique(array)

    # Crear un DataFrame con ceros
    df = pd.DataFrame(0, index=np.arange(len(array)), columns=vocabulario)

    # Llenar el DataFrame con 1s donde corresponde
    for i, palabra in enumerate(array):
        df.loc[i, palabra] = 1

    # Convertir DataFrame a un arreglo numpy
    one_hot = df.to_numpy()

    return vocabulario,one_hot

def elementos_mas_comunes(array,numPalabras):
    # Contar la frecuencia de las palabras
    frecuencia = Counter(array)

    # Obtener las 4 palabras más frecuentes
    mas_frecuentes = [palabra for palabra, _ in frecuencia.most_common(numPalabras)]

    # Eliminar el primer elemento
    if mas_frecuentes:
        mas_frecuentes.pop(0)

    # Imprimir el resultado
    print(mas_frecuentes)

    return mas_frecuentes

def coger_solo_elementos_comunes(array_recogido,palabras_frecuentes):
    y = []
    X_imagenes = []
    X_keyPoints = []

    # Filtrar elementos cuyo primer elemento está en palabras_frecuentes
    elementos_filtrados = [x for x in array_recogido if x[0][0] in palabras_frecuentes]

    # Obtener la longitud máxima de los elementos filtrados
    if elementos_filtrados:
        longitud_maxima = max(len(x) for x in elementos_filtrados)
    else:
        longitud_maxima = 0


    for i in range(len(elementos_filtrados)):
        palabra = elementos_filtrados[i][0][0]
        if palabra not in (palabras_frecuentes):
            continue
        y.append(palabra)
        elemento_imagenes = []
        elemento_keyPoints = []
        for l in range(len(elementos_filtrados[i])):
            elemento_imagenes.append(np.array(elementos_filtrados[i][l][1]))
            elemento_keyPoints.append(np.array(elementos_filtrados[i][l][2]))

        while len(elemento_imagenes) < longitud_maxima:
            elemento_imagenes.append(np.array(np.zeros_like(elemento_imagenes[0])))
            elemento_keyPoints.append(np.array(np.zeros_like(elemento_keyPoints[0])))

        X_imagenes.append(np.array(elemento_imagenes))
        X_keyPoints.append(np.array(elemento_keyPoints))

    X_imagenes = np.array(X_imagenes)
    X_keyPoints = np.array(X_keyPoints)
    y = np.array(y)

    return X_imagenes,X_keyPoints,y,longitud_maxima

def mezclar_array_con_semilla(array, semilla):
    # Crear una copia del array original para no modificar el original
    array_mezclado = array.copy()

    # Inicializar el generador de números aleatorios con la semilla
    random.seed(semilla)

    # Algoritmo de Fisher-Yates
    for i in range(len(array_mezclado) - 1, 0, -1):
        j = random.randint(0, i)
        array_mezclado[i], array_mezclado[j] = array_mezclado[j], array_mezclado[i]

    return array_mezclado


def mezclar_y_dividir(x_imagenes, x_keypoints, y, test_size=0.2, random_state=123):
    # Asegurarse de que las longitudes coincidan
    assert len(x_imagenes) == len(x_keypoints) == len(
        y), "Los arrays de datos y etiquetas deben tener la misma longitud"

    # Mezclar los datos y etiquetas manteniendo su correspondencia
    indices = np.arange(len(y))
    np.random.seed(random_state)
    np.random.shuffle(indices)

    x_imagenes = x_imagenes[indices]
    x_keypoints = x_keypoints[indices]
    y = y[indices]

    # Dividir los datos de manera estratificada para mantener la proporción de etiquetas
    x_img_train, x_img_val, x_kp_train, x_kp_val, y_train, y_val = train_test_split(
        x_imagenes, x_keypoints, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return x_img_train, x_img_val, x_kp_train, x_kp_val, y_train, y_val


def GuardarValoresROCfichero(nombreModelo, resultadosROC):
    # Definir el nombre del archivo
    nombre_archivo = f"datosROC{nombreModelo}.txt"

    # Verificar si el archivo ya existe
    if os.path.exists(nombre_archivo):
        # Si el archivo existe, eliminarlo
        os.remove(nombre_archivo)

    # Escribimos los resultados en un fichero de texto
    with open(nombre_archivo, 'w') as file:
        # Escribir los elementos en el archivo, separados por espacios
        file.write(' '.join(map(str, resultadosROC)))

##################################################
#########           Arquitectura         #########
##################################################
def cnn_model_images(input_shape, padding_value=0):
    input_image = Input(shape=input_shape, name='input_image')

    masked_image = TimeDistributed(Masking(mask_value=padding_value))(input_image)

    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(masked_image)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)

    x = TimeDistributed(Flatten())(x)

    return Model(inputs=input_image, outputs=x, name='image_cnn')

def cnn_model_keypoints(input_shape, padding_value=0):
    input_keypoints = Input(shape=input_shape, name='input_keypoints')

    masked_keypoints = Masking(mask_value=padding_value)(input_keypoints)

    y = TimeDistributed(Dense(64, activation='relu'))(masked_keypoints)
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Dropout(0.5))(y)

    y = TimeDistributed(Dense(32, activation='relu'))(y)
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Dropout(0.5))(y)

    y = TimeDistributed(Flatten())(y)

    return Model(inputs=input_keypoints, outputs=y, name='keypoints_mlp')


def cnn_model(image_input_shape, keypoints_input_shape, num_classes):
    # Crear las redes individuales
    image_cnn = cnn_model_images(image_input_shape)
    keypoints_mlp = cnn_model_keypoints(keypoints_input_shape)

    # Obtener las salidas de ambas redes
    image_features = image_cnn.output
    keypoints_features = keypoints_mlp.output

    # Concatenar las caracteristicas extraidas de las imágenes y los keypoints
    combined = Concatenate()([image_features, keypoints_features])

    # Agregar una capa bidirectional LSTM para modelar la secuencia temporal
    lstm_out = Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))(combined)

    # Capa de salida
    output = Dense(num_classes, activation='softmax', name='output')(lstm_out)

    # Definir el modelo
    model = Model(inputs=[image_cnn.input, keypoints_mlp.input], outputs=output)

    return model


#######################################################
#########              Main                   #########
#######################################################
def entrenar_red():
    print("Recogiendo datos...")

    archivo_json = ["../Materiales/data_set/recursos/datos_divididos_listos_parte_1.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_2.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_3.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_4.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_5.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_6.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_7.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_8.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_9.json","../Materiales/data_set/recursos/datos_divididos_listos_parte_10.json"]
    array_recogido = cargar_array_desde_json(archivo_json)

    print("Datos cargados")

    y = []
    X_imagenes = []
    X_keyPoints = []

    palabras = [array_recogido[i][0][0] for i in range(len(array_recogido))]

    palabras_frecuentes = elementos_mas_comunes(palabras, 3 + 1)

    X_imagenes, X_keyPoints, y,longitud_maxima = coger_solo_elementos_comunes(array_recogido, palabras_frecuentes)

    """
    for i in range(len(array_recogido)):
        y.append(array_recogido[i][0][0])
        elemento_imagenes = []
        elemento_keyPoints = []
        for l in range(len(array_recogido[i])):
            elemento_imagenes.append(np.array(array_recogido[i][l][1]))
            elemento_keyPoints.append(np.array(array_recogido[i][l][2]))

        while len(elemento_imagenes) < longitud_maxima:
            elemento_imagenes.append(np.array(np.zeros_like(elemento_imagenes[0])))
            elemento_keyPoints.append(np.array(np.zeros_like(elemento_keyPoints[0])))

        X_imagenes.append(np.array(elemento_imagenes))
        X_keyPoints.append(np.array(elemento_keyPoints))
        
    """

    print(contar_palabras(y))

    vocabulario,y = transformar_one_hot(y)

    num_classes = len(vocabulario)

    print(f"Numero de clases {num_classes}")
    print(f"Longitud maxima: {longitud_maxima}")

    image_input_shape = (longitud_maxima,32, 32,1)

    keypoints_input_shape = (longitud_maxima,12,2)

    crossValidationSplit = 10

    # CV - 10
    kf = KFold(n_splits=crossValidationSplit, shuffle=True, random_state=123)

    splitEntrenamiento = 1

    for train_index, test_index in kf.split(X_imagenes,X_keyPoints, y):
        X_train_imagenes, X_test_imagenes = X_imagenes[train_index], X_imagenes[test_index]
        X_train_keyPoints, X_test_keyPoints = X_keyPoints[train_index], X_keyPoints[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f'x_train {X_imagenes.shape} ')
        print(f'x_train {X_keyPoints.shape} ')
        print(f'y_train {y.shape} ')

        model = cnn_model(image_input_shape, keypoints_input_shape, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        #early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit([X_train_imagenes, X_train_keyPoints], y_train, batch_size=32, epochs=25, validation_data=([X_test_imagenes, X_test_keyPoints],y_test))
        #history = model.fit([X_imagenes, X_keyPoints], y, epochs=50, batch_size=32,validation_split=0.2, callbacks=[early_stopping])
        # history = model.fit(train_datagen, steps_per_epoch=len(X_train) // batch_size, epochs=epochs,
        #                    validation_data=(X_test, y_test), verbose=2)

        # Obtener las métricas del historial
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Graficar la precisión
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        # Graficar la pérdida
        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.show()

        #Visualizar datos del split
        loss, acc = model.evaluate([X_test_imagenes, X_test_keyPoints], y_test, batch_size=32)
        y_pred = model.predict([X_test_imagenes, X_test_keyPoints])
        print(y_pred)
        #resultadosROC.append(roc_auc_score(y_test, y_pred[:, 1],multi_class='ovr'))
        resultadosROC.append(roc_auc_score(y_test, y_pred, multi_class='ovr'))
        print(f"Split numero {splitEntrenamiento}:")
        print(f'loss: {loss:.2f} acc: {acc:.2f}')
        #print(f'AUC {roc_auc_score(y_test, y_pred[:, 1], ):.4f}')
        print(f'AUC {resultadosROC[splitEntrenamiento-1]:.4f}')

        print('Predictions')
        y_pred_int = y_pred.argmax(axis=1)
        print(collections.Counter(y_pred_int), '\n')

        splitEntrenamiento = splitEntrenamiento + 1

        model.save(f'{nombreModelo}.keras')

        # SHAP interpretation
        shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        indices = np.random.choice(np.arange(len(X_test_imagenes)), size=10, replace=False)
        X_explain = X_test_imagenes[indices]

        X_train_subset = X_train_imagenes[:1000]
        explainer = shap.DeepExplainer(model, X_train_subset)
        shap_values = explainer.shap_values(X_explain)

        shap.image_plot(shap_values, -X_explain)

    GuardarValoresROCfichero(nombreModelo,resultadosROC)

def main():
    #Primero cargar los datos_audio con pandas!
    #Necesitamos recoger 30 imagenes por segundo
    for ruta in rutas_imagenes:
        imagen = cv2.imread(ruta)
        if imagen is None:
            print(f"No se pudo cargar la imagen: {ruta}")

        #Analizamos la primera cara
        lips_points = extraer_lips_points(imagen)
        #Vemos el angulo
        angulo = angulo_boca(lips_points[0], lips_points[5])
        print(angulo)
        #Rotamos la imagen
        imagen_rotada = rotar_imagen_angulo_0(imagen,angulo,lips_points[0], lips_points[5])
        #Volvemos a analizar la imagen con la rotacion
        lips_points = extraer_lips_points(imagen_rotada)
        #Verificamos que el angulo es 0 y casi igual a 0 (cuando la division no es entera)
        angulo = angulo_boca(lips_points[0], lips_points[5])
        print(angulo)
        #Recortamos los labios
        imagen_labios = recortar_imagen_labios(lips_points, imagen_rotada)
        altura , anchura, channels  = imagen_labios.shape
        print(f"{altura} {anchura}")
        #La redimensionamos a 128x128 en escala de grises para ser analizada por la red
        dimension = (128,128)
        #imagen_labios_gris = redimensionar_imagen_gris(dimension,imagen_labios)
        #resize = resize_image(imagen_labios)
        resize2 = resize_and_pad(imagen_labios)

def generar_json_lips_points_datos_normalizados():
    nombre_frame_map = "../Materiales/data_set/recursos/Capitulo_1_El_Principito_prueba.srt"
    nombre_capeta_frames = "../Materiales/data_set/frames2"
    archivo_json = "../Materiales/data_set/recursos/datos_listos_2.json"
    array = leer_srt(nombre_frame_map)
    print(len(array))
    array_letras_frames = recogiendo_frames_de_cada_palabra(array, nombre_capeta_frames)
    array_letras_frames_verificados = verificar_datos(array_letras_frames, nombre_capeta_frames)
    array_lips_points = extraer_lips_points_datos(array_letras_frames_verificados, nombre_capeta_frames)
    guardar_array_en_json(array_lips_points, archivo_json)
    array_recogido = cargar_array_desde_json(archivo_json)

def generar_json_imagenes_frames_datos_normalizados():
    nombre_frame_map = "../Materiales/data_set/recursos/Capitulo_1_El_Principito_prueba.srt"
    nombre_capeta_frames = "../Materiales/data_set/frames2"
    archivo_json = "../Materiales/data_set/recursos/datos_divididos.json"
    carpeta_imagenes_labios = "../Materiales/data_set/frames_labios_gris_2"
    array = leer_srt(nombre_frame_map)
    array_letras_frames = recogiendo_frames_de_cada_palabra(array, nombre_capeta_frames)
    #array_letras_frames_verificados = verificar_datos(array_letras_frames, nombre_capeta_frames)
    array_lips_points = extraer_lips_points_datos(array_letras_frames, nombre_capeta_frames)
    print(f"Con frames {array_letras_frames[0][1][0]}")
    print(f"Con lips points {array_lips_points[0][1][0]}")

    datos_listos_para_la_red = []

    for i in range(len(array_letras_frames)):
        #palabra
        imagenes_normalizadas = []
        lips_points_normalizados = []
        for l in range(len(array_letras_frames[i][1])):
            #frame / key_point
            frame = array_letras_frames[i][1][l]
            lips_points = array_lips_points[i][1][l]
            imagen = cv2.imread(f"{nombre_capeta_frames}/{frame}")
            if imagen is None:
                print(f"No se pudo cargar la imagen: {nombre_capeta_frames}/{frame}")

            angulo = angulo_boca(lips_points[0], lips_points[5])

            imagen_rotada = rotar_imagen_angulo_0(imagen, angulo, lips_points[0], lips_points[5])
            lips_points = son_labios_detectados(imagen_rotada)

            imagen_labios = recortar_imagen_labios(lips_points, imagen_rotada)

            imagen_rezise = resize_and_pad(imagen_labios)

            imagen_normalizada = normalizar_imagenes(imagen_rezise)

            imagenes_normalizadas.append(numpy_array_to_list(imagen_normalizada))
            lips_points_normalizados.append(numpy_array_to_list(normalize_keypoints(lips_points)))

            guardar_imagen_cv2(imagen_rezise,f"{carpeta_imagenes_labios}/{frame}")

        datos_listos_para_la_red.append(numpy_array_to_list((array_letras_frames[i][0],imagenes_normalizadas,lips_points_normalizados)))

    #print(datos_listos_para_la_red)

    guardar_array_en_json(datos_listos_para_la_red, archivo_json)

def aumentado_de_datos():
    nombre_frame_map = "../Materiales/data_set/recursos/Capitulo_1_El_Principito_prueba.srt"
    nombre_capeta_frames = "../Materiales/data_set/frames_labios_gris_2"
    carpeta_aumentado_de_datos = "../Materiales/data_set/aumentado_de_datos"

    archivo_json_carga = ["../Materiales/data_set/recursos/datos_divididos_parte_1.json","../Materiales/data_set/recursos/datos_divididos_parte_2.json","../Materiales/data_set/recursos/datos_divididos_parte_3.json","../Materiales/data_set/recursos/datos_divididos_parte_4.json","../Materiales/data_set/recursos/datos_divididos_parte_5.json","../Materiales/data_set/recursos/datos_divididos_parte_6.json","../Materiales/data_set/recursos/datos_divididos_parte_7.json","../Materiales/data_set/recursos/datos_divididos_parte_8.json","../Materiales/data_set/recursos/datos_divididos_parte_9.json","../Materiales/data_set/recursos/datos_divididos_parte_10.json"]
    archivo_json_guardado = "../Materiales/data_set/recursos/datos_divididos_listos.json"
    array_recogido = cargar_array_desde_json(archivo_json_carga)

    X_imagenes = [subarray[1] for subarray in array_recogido]
    X_keyPoints = [subarray[2] for subarray in array_recogido]
    y = [subarray[0] for subarray in array_recogido]

    array = leer_srt(nombre_frame_map)
    array_letras_frames = recogiendo_frames_de_cada_palabra(array, nombre_capeta_frames)

    datos_listos_aumentado_datos = []

    print(contar_palabras(y))

    for i in range(len(array_letras_frames)):
        #palabra

        palabra = array_letras_frames[i][0]

        nuevos_datos = []

        for x in [-3, -2, -1, 0, 1, 2, 3]:
            nuevos_datos.append([])

        for l in range(len(array_letras_frames[i][1])):
            #frame / key_point
            lips_points = X_keyPoints[i][l]
            frame = array_letras_frames[i][1][l]
            imagen = cv2.imread(f"{nombre_capeta_frames}/{frame}")
            if imagen is None:
                print(f"No se pudo cargar la imagen: {nombre_capeta_frames}/{frame}")

            contador = 0
            for x in [-3,-2,-1,0,1,2,3]:
                imagen_desplazada = desplazar_imagen(imagen,x)
                imagen_gris = cv2.cvtColor(imagen_desplazada, cv2.COLOR_BGR2GRAY)
                imagen_normalizada = normalizar_imagenes(imagen_gris)

                nuevos_datos[contador].append(numpy_array_to_list((numpy_array_to_list(imagen_normalizada),numpy_array_to_list(lips_points))))

                guardar_imagen_cv2(imagen_desplazada,f"{carpeta_aumentado_de_datos}/{frame.replace(".png", f"_{contador}")}.png")
                contador = contador + 1

        for i in range(len(nuevos_datos)):
            anadir_elemeto = []
            for l in range(len(nuevos_datos[i])):
                anadir_elemeto.append((palabra, nuevos_datos[i][l][0], nuevos_datos[i][l][1]))
            datos_listos_aumentado_datos.append(numpy_array_to_list(anadir_elemeto))

    guardar_array_en_json(datos_listos_aumentado_datos, archivo_json_guardado)

def depurar_2():
    archivo_json = "../Materiales/data_set/recursos/json_labios.json"
    array_recogido = cargar_array_desde_json(archivo_json)

def cargar_datos_listo():
    archivo_json = "../Materiales/data_set/recursos/datos_listos.json"
    array_recogido = cargar_array_desde_json(archivo_json)
    print("listo")

########################################################
############         Probando modelo         ###########
########################################################
class ColaLimitada:
    def __init__(self, max_size=27):
        self.max_size = max_size
        self.cola = collections.deque(maxlen=max_size)

    def agregar(self, item):
        self.cola.append(item)  # Agrega un elemento a la cola

    def obtener_elementos(self):
        return list(self.cola)  # Devuelve una lista con los elementos actuales en la cola

    def esta_vacia(self):
        return len(self.cola) == 0  # Verifica si la cola está vacía

    def tamano(self):
        return len(self.cola)  # Devuelve el número de elementos en la cola
    
def probar_modelo_tiempo_real():
    cap = cv2.VideoCapture(0)
    cola = ColaLimitada(max_size=27)
    # Cargar el modelo desde el directorio donde se guardó
    model = tf.keras.models.load_model('3_variables.keras')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Imagen frame', frame)
        time.sleep(60)
        cola.agregar(frame)

        if cola.tamano() == 27:
            X_imagenes_normalizadas, X_keypoints_normalizados = analizar_frames(cola)
            if len(X_imagenes_normalizadas) == 0:
                continue
            data = np.array([X_imagenes_normalizadas, X_keypoints_normalizados])

            predictions = model.predict(data)

            # Supongamos que predictions[0] es la probabilidad de la clase positiva
            precision = predictions[0][0]  # Ajusta según la estructura de tus predicciones
            clase_predicha = np.argmax(predictions[0])
            threshold = 0.80

            if precision > threshold:
                # Añadir un mensaje en el video si la precisión es mayor al 80%
                cv2.putText(frame, f'{precision}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'{clase_predicha}', (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar el video en una ventana
        #cv2.imshow('Video', frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def analizar_frames(cola):
    X_imagenes_normalizadas = []
    X_keypoints_normalizados = []
    frames = cola.obtener_elementos()
    for frame in frames:
        cv2.imshow('Imagen frame', frame)
        time.sleep(10)
        lips_points = son_labios_detectados(frame)
        if len(lips_points) != 12:
            break
        angulo = angulo_boca(lips_points[0], lips_points[5])

        imagen_rotada = rotar_imagen_angulo_0(frame, angulo, lips_points[0], lips_points[5])
        cv2.imshow('Imagen frame rotada', imagen_rotada)
        time.sleep(10)
        lips_points = son_labios_detectados(imagen_rotada)

        imagen_labios = recortar_imagen_labios(lips_points, imagen_rotada)

        resize2 = resize_and_pad(imagen_labios)

        cv2.imshow('Imagen frame normalizada', resize2)
        time.sleep(10)

        imagen_normalizada = normalizar_imagenes(resize2)

        X_imagenes_normalizadas.append(numpy_array_to_list(imagen_normalizada))
        X_keypoints_normalizados.append(numpy_array_to_list(normalize_keypoints(lips_points)))

    return X_imagenes_normalizadas,X_keypoints_normalizados

def visualizar_predicciones(modelo):
    print("Recogiendo datos...")

    archivo_json = ["../Materiales/data_set/recursos/datos_divididos_listos_parte_1.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_2.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_3.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_4.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_5.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_6.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_7.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_8.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_9.json",
                    "../Materiales/data_set/recursos/datos_divididos_listos_parte_10.json"]
    array_recogido = cargar_array_desde_json(archivo_json)

    print("Datos cargados")

    y = []
    X_imagenes = []
    X_keyPoints = []

    palabras = [array_recogido[i][0][0] for i in range(len(array_recogido))]

    palabras_frecuentes = elementos_mas_comunes(palabras, 3 + 1)

    X_imagenes, X_keyPoints, y, longitud_maxima = coger_solo_elementos_comunes(array_recogido, palabras_frecuentes)

    """
    for i in range(len(array_recogido)):
        y.append(array_recogido[i][0][0])
        elemento_imagenes = []
        elemento_keyPoints = []
        for l in range(len(array_recogido[i])):
            elemento_imagenes.append(np.array(array_recogido[i][l][1]))
            elemento_keyPoints.append(np.array(array_recogido[i][l][2]))

        while len(elemento_imagenes) < longitud_maxima:
            elemento_imagenes.append(np.array(np.zeros_like(elemento_imagenes[0])))
            elemento_keyPoints.append(np.array(np.zeros_like(elemento_keyPoints[0])))

        X_imagenes.append(np.array(elemento_imagenes))
        X_keyPoints.append(np.array(elemento_keyPoints))

    """

    print(contar_palabras(y))

    vocabulario, y = transformar_one_hot(y)

    num_classes = len(vocabulario)

    print(f"Numero de clases {num_classes}")
    print(f"Longitud maxima: {longitud_maxima}")

    image_input_shape = (longitud_maxima, 32, 32, 1)

    keypoints_input_shape = (longitud_maxima, 12, 2)

    crossValidationSplit = 10

    # CV - 10
    kf = KFold(n_splits=crossValidationSplit, shuffle=True, random_state=123)

    splitEntrenamiento = 1

    for train_index, test_index in kf.split(X_imagenes, X_keyPoints, y):
        X_train_imagenes, X_test_imagenes = X_imagenes[train_index], X_imagenes[test_index]
        X_train_keyPoints, X_test_keyPoints = X_keyPoints[train_index], X_keyPoints[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(f'X_test_imagenes {X_test_imagenes.shape} ')
        print(f'X_test_keyPoints {X_test_keyPoints.shape} ')
        print(f'y_train {y.shape} ')
        # Selecciona 10 imágenes al azar de X_test
        indices = np.random.choice(np.arange(len(y_test)), size=10, replace=False)

        imagenes = X_test_imagenes[indices]
        keyPoints = X_test_keyPoints[indices]
        y = y_test[indices]

        for i in range(len(y)):
            predictions = modelo.predict([imagenes[i], keyPoints[i]])

            precision = predictions[0][0]
            clase_predicha = np.argmax(predictions[0])
            print(f"-----------------------")
            print(f"Valor esperado: {y[i]}")
            print(f"Valor predicho: {clase_predicha}, con precision de {precision}")
            print(f"Resto de precisiones: {predictions[0]}")

if __name__ == "__main__":
    #main()
    #depurar()
    #depurar_2()
    #generar_json_imagenes_frames_datos_normalizados()
    #cargar_datos_listo()
    #aumentado_de_datos()
    #entrenar_red()
    #generar_json_imagenes_frames_datos_normalizados()
    #aumentado_de_datos()
    #entrenar_red()
    direccion_modelo = r"D:\Escritorio\Universidad\5. Quinto año\Segundo Cuatri\TFG\CapturaDeLabios\3_variables_25_epocas_tiempo_real.keras"
    model = tf.keras.models.load_model(direccion_modelo)
    visualizar_predicciones(model)


