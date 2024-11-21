import cv2
import dlib
import math
import time
import os

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

# Cargar el detector de caras y el predictor de puntos clave de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Escritorio\Master\PrimerCuatri\EXTRACTION_DE_DONNEES\Red_Habla\shape_predictor_68_face_landmarks.dat")

#key_points_interesantes = [48, 49, 50, 52, 53, 54 , 55, 56, 58, 59, 62, 66]
key_points_interesantes = [48, 49, 50, 51, 52, 53, 54 , 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

# Iniciar la captura de video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

threshold = 1 # Umbral para detectar cambios significativos en los labios


def angulo_boca(comisura_labio_izq, comisura_labio_der):
    dx = comisura_labio_der[0] - comisura_labio_izq[0]
    dy = comisura_labio_der[1] - comisura_labio_izq[1]

    # Calcula el ángulo en radianes
    angulo_radianes = math.atan2(dy, dx)

    # Convierte el ángulo a grados
    angulo_grados = math.degrees(angulo_radianes)

    return angulo_grados

def normalize_keypoints(all_lips_points):
    all_lips_points = np.array(all_lips_points)
    min_vals = all_lips_points.min(axis=0)
    max_vals = all_lips_points.max(axis=0)
    normalized_points = (all_lips_points - min_vals) / (max_vals - min_vals)
    return normalized_points
    
def rotar_imagen_angulo_0(imagen,angulo,point1,point2):
    centro_punto_D1 = (point1[0]+point2[0])/2
    centro_punto_D2 = (point1[1]+point2[1])/2
    #center = (int((point1[0]+point2[0])//2),int((point1[1]+point2[1])//2))
    centro = (centro_punto_D1,centro_punto_D2)
    rotation_matrix = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    imagen_rotada = cv2.warpAffine(imagen, rotation_matrix, (imagen.shape[1], imagen.shape[0]))
    return imagen_rotada

def lips_points_labios(imagen):
    lips_points = []

    # Detectar caras
    faces = detector(imagen)

    # Para cada cara detectada, predecir los puntos clave
    for face in faces:
        landmarks = predictor(imagen, face)

        lips_points = []

        for n in key_points_interesantes:  # Los índices de los puntos de los labios en el modelo de 68 puntos
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lips_points.append((x, y))
            
        break

    return np.array(lips_points)

def dibujar_labios(lips_points,imagen):
    for tupla in lips_points:
        cv2.circle(imagen, (tupla[0], tupla[1]), 1, (0, 255, 0), -1)
        
    cv2.imshow("Imagen rotada", imagen)
    return imagen

def diferencia_frame(lips_points_anterior,lips_points_nuevo):
    return abs(lips_points_anterior-lips_points_nuevo)

def crear_grafica(historial):
    plt.plot(historial, marker='o', linestyle='-', color='b', label='Cambios entre frames')

    # Personalización de la gráfica
    
    plt.xlabel("Índice del Frame")
    plt.ylabel("Cambio")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Línea en y=0
    plt.grid(alpha=0.3)
    #plt.legend()
    
    if len(key_points_interesantes) <= 12:
        plt.title("Cambios entre Frames 12 Key Points")
        ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\12_key_points\\"
    else:  
        plt.title("Cambios entre Frames 24 Key Points")
        ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\24_key_points\\"

    # Mostrar la gráfica
    plt.show()
    
def grafica_dispersion(historial):
    # Crear un diagrama de dispersión
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(historial)), historial, c='blue', alpha=0.5, edgecolor='black')
    
    # Etiquetas y título de la gráfica
    plt.xlabel("Índice")
    plt.ylabel("Valores")
    
    if len(key_points_interesantes) <= 12:
        plt.title("Dispersión Cambios entre Frames: 12 Key Points")
        ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\12_key_points\\"
    else:  
        plt.title("Dispersión Cambios entre Frames: 24 Key Points")
        ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\24_key_points\\"
        
    # Mostrar la gráfica
    plt.show()
    
def grafica_histograma(historial):
    # Crear un histograma
    plt.figure(figsize=(10, 5))
    plt.hist(historial, bins=30, color='blue', alpha=0.7, edgecolor='black')
    
    # Etiquetas y título del histograma
    plt.xlabel("Valores")
    plt.ylabel("Frecuencia")
    
    if len(key_points_interesantes) <= 12:
        plt.title("Histograma Cambios entre Frames: 12 Key Points")
        ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\12_key_points\\"
    else:  
        plt.title("Histograma Cambios entre Frames: 24 Key Points")
        ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\24_key_points\\"

    # Mostrar el histograma
    plt.show()

def tiempo_real():
    historial_diferencias = []
    historial_puntos = []
    lips_points_anterior = np.empty(0)
    while True:
        
        # Capturar el frame de la cámara
        ret, frame = cap.read()

        # Convertir a escala de grises
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        lips_points = lips_points_labios(frame)
    
        angulo = angulo_boca(lips_points[0], lips_points[5])
        imagen_rotada = rotar_imagen_angulo_0(frame, angulo, lips_points[0], lips_points[5])
        
        lips_points_rotados = lips_points_labios(imagen_rotada)
        
        if ret:
            dibujar_labios(lips_points_rotados,imagen_rotada)
        
        lips_points = normalize_keypoints(lips_points_rotados)
        
        #lips_points = lips_points_labios(imagen_rotada)
        
        #print(normalize_keypoints(lips_points))
        print(lips_points)
        
        if ret:
            dibujar_labios(lips_points_rotados,imagen_rotada)
        
        if lips_points_anterior.size != 0:
            diferencia_entre_frames = diferencia_frame(lips_points_anterior,lips_points)
            historial_diferencias.append(diferencia_entre_frames.sum())
            
        # Romper el bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
        
        historial_puntos.append(lips_points.sum(axis=1))
        
        lips_points_anterior = lips_points
        
    #crear_grafica(historial_puntos)   
    
    crear_grafica(historial_diferencias)
    grafica_histograma(historial_diferencias)
    
    print(np.array(historial_diferencias))
        
tiempo_real()

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()