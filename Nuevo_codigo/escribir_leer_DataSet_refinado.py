import numpy as np

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

if __name__ == "__main__":
    # Datos de ejemplo: etiquetas y arrays multidimensionales
    labels = ['label3', 'label4']
    arrays = [
        np.array([[13, 14, 15], [16, 17, 18]]),  # Otro array 2D
        np.array([[19.1, 20.2], [21.3, 22.4]])  # Otro array 2D con flotantes
    ]

    # Guardar los nuevos datos al final del archivo
    guardar_entrenamiento(labels, arrays, 'entrenamiento_multidimensional.txt')

    # Leer los datos desde el archivo
    etiquetas, arrays_leidos = leer_entrenamiento('entrenamiento_multidimensional.txt')

    # Mostrar los datos leídos
    for etiqueta, array in zip(etiquetas, arrays_leidos):
        print(f"Etiqueta: {etiqueta}")
        print(f"Array:\n{array}")
        print("#####")
