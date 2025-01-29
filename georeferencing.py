import cv2
import json
import os
import numpy as np
import folium
from folium.raster_layers import ImageOverlay
from PIL import Image

def georeferencing(json_path, image_path, output_path):

    # Cargar los datos del JSON
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    # Arrays para guardar las coordenadas de los puntos de control
    image_points = np.empty((len(json_data), 2), dtype="float64")
    geo_points = np.empty((len(json_data), 2), dtype="float64")

    for control_point in json_data:
        new_point = np.array([[control_point[0], control_point[1]]]) 
        image_points = np.vstack((image_points, new_point))
        new_point = np.array([[control_point[3], control_point[2]]]) 
        geo_points = np.vstack((geo_points, new_point))

    # Calcular la transformación (homografía)
    transform_matrix, _ = cv2.findHomography(image_points, geo_points, method=cv2.RANSAC)

    # Convertir la imagen a .jpg
    img = Image.open(image_path)    
    img = img.convert("RGB")  # Elimina la transparencia (convierte a fondo blanco)
    image_path_jpg = os.path.splitext(image_path)[0] + ".jpg" # Cambia la extensión a .jpg manteniendo el mismo nombre de archivo
    img.save(image_path_jpg) # Guarda la imagen como JPG

    # Cargar la imagen
    image = cv2.imread(image_path_jpg)  
    height, width, _ = image.shape

    # Definir las esquinas de la imagen original (píxeles)
    corners = np.array([[0, 0], [0, height-1], [width-1, 0], [width-1, height-1]], dtype="float64")

    # Transformar las esquinas de la imagen a coordenadas geográficas
    geo_corners = cv2.perspectiveTransform(np.array([corners]), transform_matrix)[0]

    # Obtener las coordenadas geográficas de las 4 esquinas transformadas
    top_left = geo_corners[0]
    bottom_left = geo_corners[1]
    top_right = geo_corners[2]
    bottom_right = geo_corners[3]

    # Coordenadas de los 4 puntos límites de la imagen transformada (como bounds)
    bounds = [
        [top_left[1], top_left[0]],  # Coordenada superior izquierda
        [bottom_right[1], bottom_right[0]]  # Coordenada inferior derecha
    ]

    # Crear el mapa centrado en la imagen
    m = folium.Map(location=[(top_left[1] + bottom_right[1]) / 2, (top_left[0] + bottom_right[0]) / 2], zoom_start=16)

    # Superponer la imagen transformada en el mapa
    image_overlay = ImageOverlay(
        name="Georeferenced Image",
        image=image_path_jpg,  
        bounds=bounds,
        opacity=0.6
    )
    image_overlay.add_to(m)

    # Añadir controles
    folium.LayerControl().add_to(m)

    # Guardar y mostrar el mapa
    m.save(output_path)

georeferencing('json/control_points.json', 'images/habana.png', 'georeferenced_map.html')

