# ML-Project

## Clustering de polígonos

- `dbscan.py`: Dado los polígonos se clusterizan para encontrar aquellos que son raros (equivalente en DBSCAN a los outliers).

- `metrics.py`: Funciones para calcular las métricas que se utilizarán como _features_ de los polígonos, que después serán utilizados como entrada del modelo DBSCAN.

** Falta verificar la precisión de los polígonos outliers encontrados, con respecto a todos los polígonos existentes en cada imagen. 

## Georreferenciación

- `georeferencing.py`: Dados los puntos de control se georeferencia la imagen y se superpone en un mapa de Google Maps.

** Se está probando la función de transformación que más sirve para ajustar las imágenes a su geolocalización real.