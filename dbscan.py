import cv2
import os
import json
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import shapiro, kstest, norm, anderson
from metrics import calculate_area, calculate_circularity, calculate_compactness, calculate_perimeter, convexity_measure, side_length_variation, side_length_variance

def process_shapes(json_data, output_csv):
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Name', 'Num_Sides', 'Area', 'Perimeter', 'Circularity', 'Compactness', 'Convexity', 'CV', 'Top_Left_Corner'])

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
                top_left_corner = shape['top_left_corner']

                writer.writerow([shape_name, num_sides, area, perimeter, circularity, compactness, convexity, cv, top_left_corner])

def verifying_normal_distribution(csv_path):

    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Name', 'Top_Left_Corner']).values

    # Prueba de Shapiro-Wilk
    stat, p = shapiro(df)
    if p > 0.05:
        return True
    else:
        return False

def draw_polygon(coordinates, output_path, shape_name):
    # Crear una figura y un eje
    fig, ax = plt.subplots()
    
    # Agregar el polígono al gráfico
    polygon = Polygon(coordinates, closed=True, edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(polygon)
    
    # Ajustar los límites del gráfico
    x_coords = [point[0] for point in coordinates]
    y_coords = [point[1] for point in coordinates]
    ax.set_xlim(min(x_coords) - 10, max(x_coords) + 10)
    ax.set_ylim(min(y_coords) - 10, max(y_coords) + 10)
    ax.set_aspect('equal', adjustable='datalim')
    
    # Quitar ejes para que sea solo la figura
    ax.axis('off')
    
    # Guardar la figura como imagen
    output_file = os.path.join(output_path, f"{shape_name}.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def detect_outliers_with_dbscan(csv_path, output_path, json_data, eps=0.5, min_samples=5):
    """
    Detecta outliers en los polígonos basándose en DBSCAN y guarda los polígonos identificados como outliers en una carpeta.

    Parámetros:
        csv_path: str - Ruta del archivo CSV con los features.
        output_path: str - Carpeta donde se guardarán los polígonos outliers.
        eps: float - Máxima distancia entre dos muestras para que se consideren en el mismo vecindario.
        min_samples: int - Número mínimo de muestras para formar un clúster.
    """
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)

    # Separar los nombres de los polígonos y los features
    polygon_names = df['Name']
    features = df.drop(columns=['Name', 'Top_Left_Corner', 'Area', 'Perimeter']).values

    if verifying_normal_distribution(csv_path):
        # Estandarización de los datos
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        # Normalización de los datos
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)

    # Identificar outliers (label == -1)
    outlier_indices = np.where(labels == -1)[0]
    outlier_polygon_names = polygon_names.iloc[outlier_indices]
    outlier_polygon_names_list = outlier_polygon_names.tolist()

    # Crear carpeta de outliers si no existe
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Guardar los polígonos outliers como imágenes a la carpeta
    for shape_name, shape_data in json_data.items():
        if shape_name in outlier_polygon_names_list:
            for shape in shape_data:
                coordinates = shape['coordinates']
                draw_polygon(coordinates, output_path, shape_name)

    # Guardar los outliers_filename en un archivo JSON
    with open(f"{output_path}/outliers_polygon_names.json", "w", encoding="utf-8") as file:
        json.dump(outlier_polygon_names_list, file, indent=4)

def clustering_polygons(json_path, output_path):
    # Ruta para guardar los features de los polígonos en CSV
    features_csv = "features.csv"
    # Cargar los datos del JSON
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Extraer features y guardar en CSV
    process_shapes(json_data, features_csv)
    # Detectar outliers usando DBSCAN
    detect_outliers_with_dbscan(features_csv, output_path, json_data, eps=0.3, min_samples=5)
