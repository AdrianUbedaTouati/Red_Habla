import cv2
import dlib
import math

import numpy as np
from PIL import Image

# Cargar el detector de caras y el predictor de puntos clave de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:\Escritorio\Master\PrimerCuatri\EXTRACTION_DE_DONNEES\Red_Habla\shape_predictor_68_face_landmarks.dat")

key_points_interesantes = [48 ,49,50,52,53,54 ,55,56,58,59,62,66]

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

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

def rotar_imagen_angulo_0(imagen,angulo):
    imagen_rotada = imagen.rotate(-angulo)
    imagen_rotada.show()

while True:
    # Capturar el frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faces = detector(gray)

    # Para cada cara detectada, predecir los puntos clave
    for face in faces:
        landmarks = predictor(gray, face)

        # Extraer los puntos clave de los labios
        lips_points = []

        for n in key_points_interesantes:  # Los índices de los puntos de los labios en el modelo de 68 puntos
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lips_points.append((x, y))
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        #angulo = angulo_boca(lips_points[0],lips_points[5])
        #print(angulo)
        #rotar_imagen_angulo_0(frame,angulo)
        print(normalize_keypoints(lips_points))

    # Mostrar el frame con los puntos clave dibujados
    cv2.imshow("Lips Keypoints", frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()