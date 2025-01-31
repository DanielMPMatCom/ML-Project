# Past2Polygon: Un enfoque basado en Machine Learning para la digitalización de mapas históricos.

En este trabajo se presenta un procedimiento para la digitalización y modernización de mapas antiguos, con un enfoque particular en la ciudad de La Habana. El objetivo es transformar mapas históricos proporcionados por la Oficina del Historiador de la Ciudad de La Habana en representaciones digitales modernas, permitiendo el análisis de los cambios urbanos a lo largo del tiempo.  

Inicialmente, los mapas son preprocesados mediante el recorte de bordes, la conversión a escala de grises y la aplicación de filtros para la eliminación de ruido. Posteriormente, se eliminan las etiquetas textuales y numéricas utilizando PaddleOCR, reemplazando las áreas detectadas con el color predominante mediante K-means. Luego, se emplea una red neuronal convolucional para generar un mapa de calor que diferencia calles y bloques de edificios. Las imágenes se segmentan con un algoritmo de *Flood Fill* y, utilizando la información del mapa de calor, se identifican y extraen los bloques urbanos.  

Los bloques segmentados se convierten en polígonos, extrayendo sus vértices y aristas, y se simplifican con los algoritmos de Ramer-Douglas-Peucker (RDP) y Shapely. Para la georreferenciación, se identifican polígonos atípicos mediante DBSCAN y se comparan con bloques equivalentes en mapas actuales. Una vez encontrados los bloques correspondientes, se calculan sus centroides y se alinean los mapas históricos con los modernos mediante superposición geoespacial.  

El método propuesto permite digitalizar y estructurar mapas antiguos en un formato compatible con herramientas de análisis geográfico moderno, facilitando el estudio de la evolución urbana de La Habana y su comparación con el entorno actual. Dame este texto en ingles, y proponme un nombre cool para el paper

[References](https://docs.google.com/spreadsheets/d/1QAY1Rq8ya9ovtaUcuRYeE54Nv9QQ88MFi4rdeW3I08Y/edit?usp=sharing)
