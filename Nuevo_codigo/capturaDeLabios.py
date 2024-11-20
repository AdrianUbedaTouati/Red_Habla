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
    
    plt.savefig(ruta_graficas + os.path.basename(ruta_video).replace("mp4","png"), dpi=300) 

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
    
    # Guardar la gráfica
    nombre_archivo = "diagrama_dispersion_" + os.path.basename(ruta_video).replace(".mp4", ".png")
    plt.savefig(ruta_graficas + nombre_archivo, dpi=300)

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
    
    # Guardar el histograma
    nombre_archivo = "histograma_" + os.path.basename(ruta_video).replace(".mp4", ".png")
    plt.savefig(ruta_graficas + nombre_archivo, dpi=300)

    # Mostrar el histograma
    plt.show()
    
def grafica_histograma_por_D(historial):
    i=0
    for dimension in historial:
        
        # Crear un histograma
        plt.figure(figsize=(10, 5))
        plt.hist(dimension, bins=30, color='blue', alpha=0.7, edgecolor='black')
        
        # Etiquetas y título del histograma
        plt.xlabel("Valores")
        plt.ylabel("Frecuencia")
        
        if len(key_points_interesantes) <= 12:
            plt.title(f"Histograma Punto {key_points_interesantes[i]}cambios entre frames: 12 Key Points")
            ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\12_key_points\\"
        else:  
            plt.title(f"Histograma Punto {key_points_interesantes[i]} cambios entre frames: 24 Key Points")
            ruta_graficas = r"Nuevo_codigo\Graficas_movimientos_entre_frames\24_key_points\\"
        
        # Guardar el histograma
        nombre_archivo = f"histograma_punto_{key_points_interesantes[i]}" + os.path.basename(ruta_video).replace(".mp4", ".png")
        plt.savefig(ruta_graficas + nombre_archivo, dpi=300)

        # Mostrar el histograma
        #plt.show()
        i+=1
    
#####################################################################################################################3

def procesar_video(ruta_video):
    historial_diferencias = []
    historial_puntos = []
    
    historial_diferencias_por_dimension = []
    
    for dimension in key_points_interesantes:
        historial_diferencias_por_dimension.append([])
    
    frames_totales = 0   
    frames_sin_omitir = 0
    
    lips_points_anterior = np.empty(0)

    # Captura de video desde un archivo
    cap = cv2.VideoCapture(ruta_video)

    while True:
        # Capturar el frame del video
        ret, frame = cap.read()
        if not ret:
            break

        # Procesamiento de los puntos de los labios
        lips_points_antes_rotacion = lips_points_labios(frame)
        if len(lips_points_antes_rotacion) != len(key_points_interesantes):
            break

        indice_comisura_iz = key_points_interesantes.index(48)
        indice_comisura_der = key_points_interesantes.index(54)
        
        # Cálculo del ángulo y rotación de la imagen
        angulo = angulo_boca(lips_points_antes_rotacion[indice_comisura_iz], lips_points_antes_rotacion[indice_comisura_der])
        imagen_rotada = rotar_imagen_angulo_0(frame, angulo, lips_points_antes_rotacion[indice_comisura_iz], lips_points_antes_rotacion[indice_comisura_der])
        
        # Normalizar los puntos de los labios
        lips_points = lips_points_labios(imagen_rotada)
        
        lips_points_normalizados = normalize_keypoints(lips_points)
        
        frames_totales += 1
        
        if lips_points_anterior.size != 0:
            # Calcular la distancia promedio entre los puntos actuales y los anteriores
            distances = np.linalg.norm(lips_points - lips_points_anterior, axis=1)
            mean_distance = np.mean(distances)
            historial_diferencias.append(mean_distance)

            # Si el cambio es significativo, guarda el cuadro en el video resumen
            if mean_distance > threshold:
                dibujar_labios(lips_points,imagen_rotada)
                frames_sin_omitir += 1

        """
        # Calcular la diferencia entre los cuadros
        if lips_points_anterior.size != 0:
            diferencia_entre_frames = diferencia_frame(lips_points_anterior, lips_points_normalizados)
            diferencia_entre_frames[ indice_igual > diferencia_entre_frames ] = 0
            historial_suma_diferencias.append(diferencia_entre_frames.sum())
            error_minimo = indice_igual * lips_points_normalizados.size
            frames_totales += 1   
    
            if diferencia_entre_frames.sum() > error_minimo:
                frames_sin_omitir += 1
                dibujar_labios(lips_points,imagen_rotada)
            
            for i in range(len(diferencia_entre_frames)):
                for diferencia in diferencia_entre_frames[i]:
                    historial_diferencias_por_dimension[i].append(diferencia)
        """

        # Almacenar puntos históricos
        historial_puntos.append(lips_points_normalizados.sum(axis=1))
        lips_points_anterior = lips_points

        # Romper el bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar el video y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()
    crear_grafica(historial_diferencias)
    grafica_histograma(historial_diferencias)
    
    print(frames_totales)
    print(frames_sin_omitir)
    
    #grafica_dispersion(historial_diferencias_por_dimension)
    #grafica_histograma_por_D(historial_diferencias_por_dimension)
    # Crear las gráficas con los datos recolectados
    #crear_grafica(historial_puntos)
    #crear_grafica(historial_suma_diferencias)

def tiempo_real():
    historial_diferencias = []
    historial_puntos = []
    lips_points_anterior = np.empty(0)
    while True:
        
        # Capturar el frame de la cámara
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir a escala de grises
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        lips_points = lips_points_labios(frame)
        if len(lips_points) != 12:
            break
    
        angulo = angulo_boca(lips_points[0], lips_points[5])
        imagen_rotada = rotar_imagen_angulo_0(frame, angulo, lips_points[0], lips_points[5])
        
        lips_points = normalize_keypoints(lips_points_labios(imagen_rotada))
        
        #lips_points = lips_points_labios(imagen_rotada)
        
        #print(normalize_keypoints(lips_points))
        print(lips_points)
        
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
        
# Ejemplo de uso
ruta_video_hola_lento = r"Nuevo_codigo\Videos\lento_buenos_dias.mp4"
ruta_video_hola_rapido = r"Nuevo_codigo\Videos\rapido_buenos_dias.mp4"
ruta_video_quieto_largo = r"Nuevo_codigo\Videos\quieto_largo.mp4"
ruta_video_quieto_rapido = r"Nuevo_codigo\Videos\quieto_rapido.mp4"

ruta_video = ruta_video_quieto_rapido
procesar_video(ruta_video)
#tiempo_real()


# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()