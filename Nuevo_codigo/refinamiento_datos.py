import cv2
import os
import math
import numpy as np

def lips_points_labios(frame):
    """
    Función ficticia que simula la detección de puntos clave en los labios.
    Implementa esta función según las necesidades reales de procesamiento.
    """
    # Simulando algunos puntos clave (reemplazar con detección real)
    return np.random.rand(20, 2)  # 20 puntos clave con coordenadas x, y

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

# Ejemplo de uso
video_path = r"DataSet\s1\s1\bbaf2n.mpg"
intervals_file = r"DataSet\alignments\alignments\s1\bbaf2n.align"
output_dir = r"Lips_Points_refinado"

print(split_video_to_labels_and_keypoints(video_path, intervals_file))