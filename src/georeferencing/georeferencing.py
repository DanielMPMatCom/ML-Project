import cv2
import json
import os
import numpy as np
import folium
from folium import Element
from folium.raster_layers import ImageOverlay
from PIL import Image

def georeferencing(json_path, images_path, output_path):
    """
    Georreferencia y superpone imágenes en un mapa interactivo de Folium.

    Esta función toma un conjunto de puntos de control desde un archivo JSON, calcula una 
    transformación de homografía para alinear imágenes con coordenadas geográficas y 
    superpone múltiples imágenes en un mapa de Folium. 

    Parámetros:
    ----------
    json_path : str
        Ruta al archivo JSON que contiene los puntos de control.
        Cada punto de control debe estar en formato: [x_imagen, y_imagen, latitud, longitud].
    
    images_path : list of str
        Lista de rutas de las imágenes a georreferenciar.
    
    output_path : str
        Ruta donde se guardará el archivo HTML con el mapa resultante.

    """

    # Cargar los datos del JSON
    with open(json_path, 'r') as file:
        json_data = json.load(file)

    image_points = np.array([[cp[0], cp[1]] for cp in json_data], dtype="float64")
    geo_points = np.array([[cp[3], cp[2]] for cp in json_data], dtype="float64")

    # Calcular la transformación (homografía)
    transform_matrix, _ = cv2.findHomography(image_points, geo_points, method=cv2.RANSAC)

    for image_path in images_path:
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

    for idx, image_path in enumerate(images_path):
        layer_name = os.path.basename(image_path)  # Nombre de la capa basado en el nombre del archivo
        feature_group = folium.FeatureGroup(name=layer_name)  # Crear grupo de capas

        # Superponer la imagen transformada en el mapa
        image_overlay = ImageOverlay(
            image=os.path.splitext(image_path)[0] + ".jpg",  
            bounds=bounds,
            opacity=0.6
        )
        image_overlay.add_to(feature_group)
        feature_group.add_to(m)

        # Agregar el slider para controlar opacidad con JavaScript
        opacity_control = Element(f"""
            <div style="position: fixed; 
                        bottom: {10 + idx * 60}px; left: 50px; width: 200px; height: 40px; 
                        background-color: white; z-index:9999; padding: 10px; 
                        border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
                <label for="opacity-slider-{idx}"> Opacidad ({layer_name}):</label>
                <input id="opacity-slider-{idx}" type="range" min="0" max="1" step="0.1" value="0.6" 
                       oninput="document.getElementById('overlay-{idx}').style.opacity = this.value;">
            </div>
            <script>
                setTimeout(function(){{
                    let imgOverlay = document.querySelectorAll('img.leaflet-image-layer')[{idx}];
                    if (imgOverlay) {{
                        imgOverlay.id = 'overlay-{idx}';
                    }}
                }}, 500);
            </script>
        """)
        m.get_root().html.add_child(opacity_control)

    # Añadir controles
    folium.LayerControl().add_to(m)

    # Guardar y mostrar el mapa
    m.save(output_path)

# Ejemplo de cómo llamarla
georeferencing('json/control_points.json', ['images/habana.png', 'images/image_1.jpg'], 'georeferenced_map.html')

