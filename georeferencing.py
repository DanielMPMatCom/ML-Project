import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import folium
from folium.raster_layers import ImageOverlay
from PIL import Image

# Puntos de control
x1 = [5044, 2284, 23.143211, -82.378658]  # Infanta y 23 (La primera de abajo))
x2 = [344, 2276, 23.140061, -82.350725] # Esquina de Mercaderes y O'Reilly (La primera de abajo)
x3 = [2268, 3524, 23.129823, -82.351820] # Esquina de Compostela y San Isidro (La segunda de abajo)
x4 = [2480, 892, 23.124520, -82.374515] # Esquina de Infanta y Santa Marta (La segunda de arriba)


# Puntos de control: (x, y) en la imagen y (lat, lon) en el mapa
image_points = np.array([
    [x1[0], x1[1]],
    [x2[0], x2[1]],
    [x3[0], x3[1]],
    [x4[0], x4[1]]  # Añade tantos puntos como tengas
], dtype="float64")

geo_points = np.array([
    [x1[3], x1[2]],
    [x2[3], x2[2]],
    [x3[3], x3[2]],
    [x4[3], x4[2]]  # Asegúrate de que coincidan con image_points
], dtype="float64")

# Calcular la transformación (homografía)
transform_matrix, _ = cv2.findHomography(image_points, geo_points, method=cv2.RANSAC)

# Leer la imagen original
image = cv2.imread("habana.png")  
height, width, _ = image.shape

# Transformar la imagen al sistema geográfico
geo_image = cv2.warpPerspective(image, transform_matrix, (width, height))

# Guardar la imagen georreferenciada
cv2.imwrite("geo_image.png", geo_image)

# Definir los límites geográficos
min_lon, max_lon = float(min(geo_points[:, 0])), float(max(geo_points[:, 0]))
min_lat, max_lat = float(min(geo_points[:, 1])), float(max(geo_points[:, 1]))

# Coordenadas aproximadas del bounding box de la imagen transformada
bounds = [
    [min_lat, min_lon],  # Coordenada superior izquierda
    [max_lat, max_lon]   # Coordenada inferior derecha
]

# Crear el mapa centrado en la imagen
m = folium.Map(location=[(min_lat + max_lat) / 2, (min_lon + max_lon) / 2], zoom_start=16)

# Superponer la imagen transformada en el mapa
image_overlay = ImageOverlay(
    name="Georeferenced Image",
    image="geo_image.png",  # Imagen transformada
    bounds=bounds,
    opacity=0.6
)
image_overlay.add_to(m)

# Añadir controles
folium.LayerControl().add_to(m)

# Guardar y mostrar el mapa
m.save("georeferenced_map.html")