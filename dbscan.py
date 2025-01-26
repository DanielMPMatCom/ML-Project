import cv2
import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import shapiro, kstest, norm, anderson

def calculate_features(contour):
    """
    Calcula las características geométricas de un contorno.
    
    Parámetros:
        contour: np.ndarray - Contorno detectado en la imagen.
        
    Retorna:
        list - Lista con las características calculadas.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h != 0 else 0
    compactness = area / (w * h) if w * h != 0 else 0
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return [area, perimeter, aspect_ratio, compactness, circularity]

def extract_features_from_folder(folder_path, output_csv):
    """
    Procesa todas las imágenes en una carpeta, calcula sus features y las guarda en un CSV.
    
    Parámetros:
        folder_path: str - Ruta de la carpeta con imágenes.
        output_csv: str - Ruta del archivo CSV donde se guardarán los resultados.
    """
    # Lista para almacenar los resultados
    data = []

    # Recorrer todas las imágenes en la carpeta
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png')):
            # Leer la imagen
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Verificar que la imagen se cargó correctamente
            if image is None:
                print(f"No se pudo cargar la imagen: {filename}")
                continue

            # Binarizar la imagen
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            # Encontrar contornos
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Calcular features para cada contorno
            for contour in contours:
                features = calculate_features(contour)
                data.append([filename] + features)

    # Crear un DataFrame de los resultados
    columns = ["filename", "area", "perimeter", "aspect_ratio", "compactness", "circularity"]
    df = pd.DataFrame(data, columns=columns)

    # Guardar en un archivo CSV
    df.to_csv(output_csv, index=False)
    print(f"Features guardados en: {output_csv}")

def detect_outliers_with_dbscan(csv_path, output_folder, images_folder_path, eps=0.5, min_samples=5):
    """
    Detecta outliers en las imágenes basándose en DBSCAN y guarda las imágenes identificadas como outliers en una carpeta.

    Parámetros:
        csv_path: str - Ruta del archivo CSV con los features.
        output_folder: str - Carpeta donde se guardarán las imágenes outliers.
        images_folder_path: str - Carpeta donde se encuentran las imagenes.
        eps: float - Máxima distancia entre dos muestras para que se consideren en el mismo vecindario.
        min_samples: int - Número mínimo de muestras para formar un clúster.
    """
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)

    # Separar los nombres de archivo y los features
    filenames = df['filename']
    features = df.drop(columns=['filename']).values

    # Normalizacion de los datos
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)

    # Identificar outliers (label == -1)
    outlier_indices = np.where(labels == -1)[0]
    outlier_filenames = filenames.iloc[outlier_indices]

    # Crear carpeta de outliers si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Copiar las imágenes outliers a la carpeta
    for filename in outlier_filenames:
        source_path = os.path.join(images_folder_path, filename)
        destination_path = os.path.join(output_folder, filename)
        if os.path.exists(source_path):
            cv2.imwrite(destination_path, cv2.imread(source_path))

    print(f"Nombres de las imágenes outliers: {list(outlier_filenames)}")
    print(f"Se encontraron {len(outlier_indices)} imágenes outliers.")

def verifying_normal_distribution(csv_path):

    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)
    df = df.iloc[:, 1:] 

    print('\n-------------------')
    # Prueba de Shapiro-Wilk
    stat, p = shapiro(df)
    print(f"Estadístico: {stat}, p-valor: {p}")
    if p > 0.05:
        print("Los datos parecen seguir una distribución normal.")
    else:
        print("Los datos NO siguen una distribución normal.")

    print('\n-------------------')
    # Prueba de Kolmogorov-Smirnov
    stat, p = kstest(df, "norm", args=(df.mean(), df.std()))
    print(f"Estadístico: {stat}, p-valor: {p}")
    if p > 0.05:
        print("Los datos parecen seguir una distribución normal.")
    else:
        print("Los datos NO siguen una distribución normal.")

    print('\n-------------------')
    # Prueba de Anderson
    result = anderson(df, dist='norm')
    print(f"Estadístico: {result.statistic}")
    print("Valores críticos:", result.critical_values)
    if result.statistic < result.critical_values[2]:  # Generalmente usamos el nivel de significancia del 5%
        print("Los datos parecen seguir una distribución normal.")
    else:
        print("Los datos NO siguen una distribución normal.")


if __name__ == '__main__':
    
    # Ruta de la carpeta con imágenes
    folder_path = "images"
    # Ruta del archivo CSV de salida
    output_csv = "features_output.csv"
    # Carpeta para guardar outliers
    outliers_folder = "outliers"
    
    # Extraer features y guardar en CSV
    extract_features_from_folder(folder_path, output_csv)
    
    # Detectar outliers usando DBSCAN
    detect_outliers_with_dbscan(output_csv, outliers_folder, images_folder_path = folder_path, eps=0.3, min_samples=5)

    # Comprobando si los datos tienen una distribucion normal
    # verifying_normal_distribution(output_csv)