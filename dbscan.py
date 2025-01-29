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
from sklearn.preprocessing import MinMaxScaler
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

def detect_outliers_with_dbscan(csv_path, output_folder, json_data, eps=0.5, min_samples=5):
    """
    Detecta outliers en los polígonos basándose en DBSCAN y guarda los polígonos identificados como outliers en una carpeta.

    Parámetros:
        csv_path: str - Ruta del archivo CSV con los features.
        output_folder: str - Carpeta donde se guardarán los polígonos outliers.
        eps: float - Máxima distancia entre dos muestras para que se consideren en el mismo vecindario.
        min_samples: int - Número mínimo de muestras para formar un clúster.
    """
    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)

    # Separar los nombres de archivo y los features
    filenames = df['Name']
    # features_list_name = []
    # for row in df.itertuples():
    #     if row.Convexity < 0.8:
    #         features_list_name.append(row.Name)
    features = df.drop(columns=['Name', 'Top_Left_Corner', 'Area', 'Perimeter']).values

    # Normalizacion de los datos
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features)

    # Identificar outliers (label == -1)
    outlier_indices = np.where(labels == -1)[0]
    outlier_filenames = filenames.iloc[outlier_indices]
    outlier_filenames_list = outlier_filenames.tolist()

    # Crear carpeta de outliers si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # (Esto es no va)
    # Dibujar los poligonos con menos de 0.80 de relacion
    # for shape_name, shape_data in json_data.items():
    #     if shape_name in features_list_name:
    #         for shape in shape_data:
    #             coordinates = shape['coordinates']
    #             draw_polygon(coordinates, output_folder, shape_name)

    # Guardar los polígonos outliers como imágenes a la carpeta
    for shape_name, shape_data in json_data.items():
        if shape_name in outlier_filenames_list:
            for shape in shape_data:
                coordinates = shape['coordinates']
                draw_polygon(coordinates, output_folder, shape_name)

    print(f"Se encontraron {len(outlier_indices)} imágenes outliers.")

def verifying_normal_distribution(csv_path):

    # Cargar el archivo CSV
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['Name', 'Top_Left_Corner']).values

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
    
    # Ruta al archivo JSON
    json_path = "json/001_grouped_shapefiles_rdp.json"
    # Ruta del archivo CSV de salida
    features_csv = "features.csv"
    # Carpeta para guardar outliers
    outliers_folder = "outliers"

    # Load JSON data
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Extraer features y guardar en CSV
    process_shapes(json_data, features_csv)
    print(f"Features successfully saved to {features_csv}.")
    
    # Detectar outliers usando DBSCAN
    detect_outliers_with_dbscan(features_csv, outliers_folder, json_data, eps=0.3, min_samples=5)

    # Comprobando si los datos tienen una distribucion normal
    # verifying_normal_distribution(features_csv)
