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

#############################################################
# Extraccion de datos

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

def angulo_boca(comisura_labio_izq, comisura_labio_der):
    dx = comisura_labio_der[0] - comisura_labio_izq[0]
    dy = comisura_labio_der[1] - comisura_labio_izq[1]

    # Calcula el ángulo en radianes
    angulo_radianes = math.atan2(dy, dx)

    # Convierte el ángulo a grados
    angulo_grados = math.degrees(angulo_radianes)

    return angulo_grados

def rotar_imagen_angulo_0(imagen,angulo,point1,point2):
    centro_punto_D1 = (point1[0]+point2[0])/2
    centro_punto_D2 = (point1[1]+point2[1])/2
    #center = (int((point1[0]+point2[0])//2),int((point1[1]+point2[1])//2))
    centro = (centro_punto_D1,centro_punto_D2)
    rotation_matrix = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    imagen_rotada = cv2.warpAffine(imagen, rotation_matrix, (imagen.shape[1], imagen.shape[0]))
    return imagen_rotada

#############################################################
# Normalizacion

def normalize_keypoints(all_lips_points):
    all_lips_points = np.array(all_lips_points)
    min_vals = all_lips_points.min(axis=0)
    max_vals = all_lips_points.max(axis=0)
    normalized_points = (all_lips_points - min_vals) / (max_vals - min_vals)
    return normalized_points
    

#############################################################
# Graficas
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
    
def dibujar_labios(lips_points,imagen):
    for tupla in lips_points:
        cv2.circle(imagen, (tupla[0], tupla[1]), 1, (0, 255, 0), -1)
        
    cv2.imshow("Imagen rotada", imagen)
    return imagen

#####################################################################################################################
# Manejo DataSet

#############################################################
# Extraer Caracteristicas 

def split_video_to_labels_and_keypoints(video_path, intervals_file):
    # Leer el archivo de intervalos
    with open(intervals_file, 'r') as file:
        lines = file.readlines()

    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        raise Exception("El video no se ha podido abrir correctamente")

    # Lista para almacenar los datos estructurados
    labels = []
    labels_key_points = []
    
    antiguo_label = None

    for line in lines:
        start_frame, end_frame, label = line.strip().split()
        start_frame, end_frame = math.ceil(int(start_frame) / 1000), math.ceil(int(end_frame) / 1000)

        # Establecer la posición inicial del frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        print(f"Procesando frames para la etiqueta '{label}' desde el frame {start_frame} hasta {end_frame}...")
        
        if antiguo_label != label:
            labels.append(label)
            if antiguo_label != None:
                labels_key_points.append(np.array(key_points))
            key_points = []
            antiguo_label = label
            
        # Recorrer los frames del intervalo
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                raise Exception(f"El video {video_path} no tiene el número de frames esperado.")

            key_points.append(np.array(lips_points_labios(frame)))

    # Liberar recursos de OpenCV
    cap.release()
    print("Procesamiento completado.")

    # Convertir la lista en un np.array estructurado
    return (np.array(labels),labels_key_points)
    
#############################################################
# Guardar leer Data_Set Refinado

def guardar_entrenamiento(labels, arrays, nombre_archivo):
    """
    Guarda las etiquetas y los arrays multidimensionales en un archivo de texto,
    agregándolos al final del archivo si ya existe.
    
    Args:
    labels (list): Lista de etiquetas.
    arrays (list): Lista de arrays de NumPy multidimensionales.
    nombre_archivo (str): Nombre del archivo donde se guardarán los datos.
    """
    try:
        with open(nombre_archivo, 'a') as f:  # Modo de adición ('a')
            for label, arr in zip(labels, arrays):
                f.write(f"{label}\n")  # Escribir la etiqueta
                np.savetxt(f, arr, delimiter=' ', fmt='%.6g')  # Guardar el array
                f.write("\n#####\n")  # Separador entre bloques
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        
        
# Función para leer etiquetas y arrays desde un archivo
def leer_entrenamiento(nombre_archivo):
    """
    Lee las etiquetas y los arrays multidimensionales desde un archivo de texto.
    
    Args:
    nombre_archivo (str): Nombre del archivo desde donde se leerán los datos.
    
    Returns:
    tuple:
        - labels (list): Lista de etiquetas.
        - arrays (list): Lista de arrays de NumPy multidimensionales.
    """
    labels = []
    arrays = []
    try:
        with open(nombre_archivo, 'r') as f:
            while True:
                label = f.readline().strip()  # Leer la etiqueta
                if not label:  # Si no hay más etiquetas, terminar
                    break
                arr = []
                while True:
                    line = f.readline().strip()
                    if line == '#####':  # Fin del bloque de array
                        break
                    if line:  # Evitar líneas vacías
                        arr.append(list(map(float, line.split())))  # Convertir a flotantes
                if arr:  # Verificar que el array no esté vacío
                    arr = np.array(arr)  # Convertir la lista de listas en un array de NumPy
                else:
                    arr = np.empty((0, 0))  # Array vacío si no hay datos
                labels.append(label)
                arrays.append(arr)
    except Exception as e:
        print(f"Error al leer los datos: {e}")

    return labels, arrays 
    
#####################################################################################################################
# Ejecucion expetimental, estudio de la varianza de frame y treshold

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
    
#####################################################################################################################
# Ejecucion expetimental, estudio de la varianza de frame y treshold
    
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