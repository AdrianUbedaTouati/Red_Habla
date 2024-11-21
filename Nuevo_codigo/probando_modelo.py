import cv2
import dlib
import numpy as np
import tensorflow as tf
from collections import deque

key_points_interesantes = [48, 49, 50, 51, 52, 53, 54 , 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

word_to_index = {
    np.str_('a'): 0,
    np.str_('again'): 1,
    np.str_('at'): 2,
    np.str_('b'): 3,
    np.str_('bin'): 4,
    np.str_('blue'): 5,
    np.str_('by'): 6,
    np.str_('c'): 7,
    np.str_('d'): 8,
    np.str_('e'): 9,
    np.str_('eight'): 10,
    np.str_('f'): 11,
    np.str_('five'): 12,
    np.str_('four'): 13,
    np.str_('g'): 14,
    np.str_('green'): 15,
    np.str_('h'): 16,
    np.str_('i'): 17,
    np.str_('in'): 18,
    np.str_('j'): 19,
    np.str_('k'): 20,
    np.str_('l'): 21,
    np.str_('lay'): 22,
    np.str_('m'): 23,
    np.str_('n'): 24,
    np.str_('nine'): 25,
    np.str_('now'): 26,
    np.str_('o'): 27,
    np.str_('one'): 28,
    np.str_('p'): 29,
    np.str_('place'): 30,
    np.str_('please'): 31,
    np.str_('q'): 32,
    np.str_('r'): 33,
    np.str_('red'): 34,
    np.str_('s'): 35,
    np.str_('set'): 36,
    np.str_('seven'): 37,
    np.str_('sil'): 38,
    np.str_('six'): 39,
    np.str_('soon'): 40,
    np.str_('sp'): 41,
    np.str_('t'): 42,
    np.str_('three'): 43,
    np.str_('two'): 44,
    np.str_('u'): 45,
    np.str_('v'): 46,
    np.str_('white'): 47,
    np.str_('with'): 48,
    np.str_('x'): 49,
    np.str_('y'): 50,
    np.str_('z'): 51,
    np.str_('zero'): 52,
}

index_to_word = {v: k for k, v in word_to_index.items()}

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

def normalize_keypoints_frame(points):
    """
    Normaliza un conjunto de keypoints en un frame centrando por el centroide
    y escalando por la distancia máxima desde el centroide.
    
    Args:
    - points (numpy array): Array de tamaño (K, 2), donde:
        K: Número de keypoints en un frame.
        2: Coordenadas (x, y).
    
    Returns:
    - normalized_points (numpy array): Keypoints normalizados en el rango [-1, 1].
    """
    points = np.array(points)  # Asegurar que sea un array de NumPy

    # 1. Centrar por el centroide
    centroid = points.mean(axis=0)  # Centroide del frame
    centered_points = points - centroid

    # 2. Escalar por la distancia máxima desde el centroide
    max_distance = np.linalg.norm(centered_points, axis=1).max()  # Distancia máxima al centroide
    if max_distance == 0:
        max_distance = 1e-8  # Evitar divisiones por cero
    
    scaled_points = centered_points / max_distance  # Normalizar al rango [-1, 1]
    
    return scaled_points

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo_clasificador_lstm.h5')

# Cargar el detector de caras y el predictor de puntos faciales de dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Necesitas este archivo

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Configurar el buffer (por ejemplo, para almacenar 10 frames)
N = 10  # Número de frames en el buffer
frame_buffer = deque(maxlen=N)  # Buffer circular para almacenar los últimos N frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises para la detección de caras
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    lip_landmarks = normalize_keypoints_frame(lips_points_labios(frame))
    
    # Agregar el frame actual al buffer
    frame_buffer.append(lip_landmarks)  # Agregar el frame al buffer

    # Si el buffer está lleno (y tienes suficientes frames para la secuencia)
    if len(frame_buffer) == N:
        # Convertir el buffer en una secuencia de entrada
        input_data = np.array(frame_buffer)  # Forma: (N, 20, 2)
        input_data = input_data.reshape(1, N, 20, 2)  # Forma: (1, N, 20, 2)

        # Realizar la predicción con el modelo
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction)

        # Mostrar la predicción en la pantalla
        cv2.putText(frame, f'Prediccion: {index_to_word[predicted_class]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el video en tiempo real
    cv2.imshow('Video en Tiempo Real', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()