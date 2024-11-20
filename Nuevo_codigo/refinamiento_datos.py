import os
import subprocess

def dividir_video_por_segmentos(video_path, texto_path, output_folder):
    """
    Divide un video en segmentos según un archivo de texto que especifica los tiempos y nombres de los segmentos.

    Args:
        video_path (str): Ruta al archivo de video.
        texto_path (str): Ruta al archivo de texto con los segmentos (inicio, fin, etiqueta).
        output_folder (str): Carpeta donde se guardarán los segmentos generados.

    Returns:
        list: Lista de rutas de los segmentos generados.
    """
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Leer el fichero de texto
    with open(texto_path, 'r') as file:
        lines = file.readlines()
    
    segment_paths = []
    
    for line in lines:
        try:
            start, end, label = line.strip().split()
            start_time = int(start) / 1000  # Convertir a segundos
            end_time = int(end) / 1000     # Convertir a segundos
            
            # Ruta del segmento
            output_path = os.path.join(output_folder, f"{label}.mp4")
            
            # Comando FFmpeg para extraer el segmento
            command = [
                "ffmpeg",
                "-i", video_path,
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c", "copy",  # Copiar los codecs (más rápido)
                output_path
            ]
            
            # Ejecutar el comando
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            segment_paths.append(output_path)
        
        except ValueError:
            print(f"Error al procesar la línea: {line.strip()}")

    return segment_paths

# Ejemplo de uso
video = r"DataSet\s1\s1\bbaf2n.mpg"
texto = r"DataSet\alignments\alignments\s1\bbaf2n.align"
salida = "Lips_Points_DataSet"

segmentos_generados = dividir_video_por_segmentos(video, texto, salida)
print("Segmentos generados:", segmentos_generados)
