import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import shapiro
from georeferencing.outliers_detection.metrics import calculate_area, calculate_circularity, calculate_compactness, calculate_perimeter, convexity_measure, side_length_variation
from skopt import gp_minimize
from skopt.space import Real, Integer

def process_shapes(json_data, output_csv):
    """
    Procesa los datos de polígonos contenidos en un archivo JSON, calculando diversas propiedades geométricas
    y guardándolas en un archivo CSV.

    Parámetros:
        json_data (dict): Datos de los polígonos en formato JSON.
        output_csv (str): Ruta del archivo CSV donde se guardarán los resultados.

    Este método calcula las siguientes propiedades geométricas para cada polígono:
        - Número de lados ('Num_Sides')
        - Área ('Area')
        - Perímetro ('Perimeter')
        - Circularidad ('Circularity')
        - Compacidad ('Compactness')
        - Convexidad ('Convexity')
        - Coeficiente de variación de los lados ('CV')
    """
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Name', 'Num_Sides', 'Area', 'Perimeter', 'Circularity', 'Compactness', 'Convexity', 'CV'])

        for shape_name, shape_data in json_data.items():
            for shape in shape_data:
                coordinates = shape['coordinates']
                num_sides = len(coordinates)
                area = calculate_area(coordinates)
                perimeter = calculate_perimeter(coordinates)
                compactness = calculate_compactness(area, perimeter)
                circularity = calculate_circularity(area, perimeter)
                convexity = convexity_measure(coordinates)
                cv = side_length_variation(coordinates)

                writer.writerow([shape_name, num_sides, area, perimeter, circularity, compactness, convexity, cv])

def verifying_normal_distribution(csv_path):
    """
    Verifica si los datos en un archivo CSV siguen una distribución normal utilizando la prueba de Shapiro-Wilk.

    Parámetros:
        csv_path (str): Ruta del archivo CSV con los datos.

    Retorna:
        bool: True si los datos siguen una distribución normal (p-value > 0.05), False en caso contrario.
    """
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Name']).values

    # Prueba de Shapiro-Wilk
    stat, p = shapiro(df)
    if p > 0.05:
        return True
    else:
        return False

def load_and_preprocess_data(csv_path):
    """
    Carga los datos desde un CSV, separa los nombres de los polígonos y extrae los features.
    Aplica estandarización si los datos son normales, de lo contrario, aplica normalización.

    Parámetros:
        csv_path: str - Ruta del archivo CSV con los features.

    Retorna:
        features: np.array - Datos preprocesados.
        polygon_names: pd.Series - Nombres de los polígonos.
    """
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)
    df = df[df['Num_Sides'] >= 12]

    # Separar los nombres de los polígonos y los features
    polygon_names = df['Name']
    features = df.drop(columns=['Name', 'Area', 'Perimeter']).values

    if verifying_normal_distribution(csv_path):
        # Estandarización de los datos
        scaler = StandardScaler()
    else:
        # Normalización de los datos
        scaler = MinMaxScaler()
    
    features = scaler.fit_transform(features)
    
    return features, polygon_names

def detect_outliers_with_dbscan(features, polygon_names, eps=0.5, min_samples=5):
    """
    Aplica DBSCAN para detectar outliers en los polígonos.

    Parámetros:
        features: np.array - Datos preprocesados.
        polygon_names: pd.Series - Nombres de los polígonos.
        eps: float - Máxima distancia entre dos muestras para formar un cluster.
        min_samples: int - Número mínimo de muestras para que un punto no sea outlier.

    Retorna:
        int - Número de outliers detectados.
        list - Lista de nombres de polígonos detectados como outliers.
    """
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)  

    # Identificar outliers (label == -1)
    outlier_indices = np.where(labels == -1)[0]
    outlier_polygon_names = polygon_names.iloc[outlier_indices]
    
    return len(outlier_indices), outlier_polygon_names.tolist()

def bayesian_optimization(features, polygon_names):
    """
    Realiza optimización bayesiana para encontrar los mejores valores de los hiperparámetros
    'eps' y 'min_samples' para el algoritmo DBSCAN.

    Parámetros:
        features (np.array): Datos preprocesados que representan los polígonos.
        polygon_names (pd.Series): Nombres de los polígonos.

    Retorna:
        tuple: El mejor valor de 'eps' y 'min_samples' encontrados durante la optimización bayesiana.
    """
    space = [Real(0.1, 1.0, name='eps'), Integer(2, 20, name='min_samples')]

    def objective_function(params):
        eps, min_samples = params
        num_outliers, _ = detect_outliers_with_dbscan(features, polygon_names, eps=eps, min_samples=min_samples)
        return -num_outliers  # Maximizar outliers

    res = gp_minimize(objective_function, space, n_calls=15, random_state=42)
    return res.x[0], res.x[1]

def save_outliers(output_path, outlier_names):
    """
    Guarda los nombres de los polígonos identificados como outliers en un archivo JSON.

    Parámetros:
        output_path (str): Ruta del archivo donde se guardarán los nombres de los outliers.
        outlier_names (list): Lista con los nombres de los polígonos outliers.
    """
    # Asegurar que el directorio padre existe
    output_dir = os.path.dirname(output_path) 
    # Crear una carpeta outliers si no existe  
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar los outliers en un archivo JSON
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(outlier_names, file, indent=4)

def clustering_polygons(json_path, output_path):
    """
    Realiza el proceso completo de detección de outliers en polígonos mediante DBSCAN:
    - Carga los datos desde un archivo JSON.
    - Extrae y guarda las características de los polígonos en un archivo CSV.
    - Ejecuta la optimización bayesiana para encontrar los mejores hiperparámetros para DBSCAN.
    - Detecta los outliers utilizando DBSCAN.
    - Guarda los nombres de los polígonos outliers en un archivo JSON.

    Parámetros:
        json_path (str): Ruta del archivo JSON con los datos de los polígonos.
        output_path (str): Ruta donde se guardará el archivo con los nombres de los polígonos outliers.
    """
    # Ruta para guardar los features de los polígonos en CSV
    features_csv = "features.csv"
    # Cargar los datos del JSON
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Extraer features y guardar en CSV
    process_shapes(json_data, features_csv)
    # Preprocesar datos
    features, polygon_names = load_and_preprocess_data(features_csv)
    # Ejecutar optimización bayesiana para hallar los mejores hiperparámetros para DBSCAN
    eps, min_samples = bayesian_optimization(features, polygon_names)
    # Aplicar DBSCAN
    _, outlier_names = detect_outliers_with_dbscan(features, polygon_names, eps, min_samples)

    # Guardar los outliers
    save_outliers(output_path, outlier_names)

# Ejemplo de cómo llamarla
# clustering_polygons('json/001_grouped_shapefiles_rdp.json', 'outliers/outliers_polygon_names.json')
