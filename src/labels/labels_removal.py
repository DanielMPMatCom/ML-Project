import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(pixels, n_clusters=3):
    if len(pixels) == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(pixels)
    
    # Obtener el color más representativo (centroide más grande)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = kmeans.cluster_centers_[unique[np.argmax(counts)]]
    
    return dominant_color.astype(np.uint8)

def remove_text_with_surrounding_color(image_path, polygons):
    image = cv2.imread(image_path)
    image_copy = image.copy()
    
    for polygon in polygons:
        # Crear la máscara para el polígono
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        k=8
        # Encontrar los píxeles fuera del polígono en un radio k
        dilated_mask = cv2.dilate(mask, np.ones((2*k+1, 2*k+1), np.uint8), iterations=1)
        border_mask = cv2.bitwise_xor(dilated_mask, mask)
        
        # Obtener los píxeles dentro del polígono
        # polygon_pixels = image[np.where(mask == 255 or border_mask == 255)]
        polygon_pixels = image[np.where((mask == 255) | (border_mask == 255))]

        
        if len(polygon_pixels) > 0:
            # Calcular el color predominante (asumiendo que ya tienes esta función)
            predominant_color = get_dominant_color(polygon_pixels)
            
            # Aplicar el color dentro del polígono
            eroded_mask = cv2.erode(mask, np.ones((2*1+1, 2*1+1), np.uint8), iterations=1)
            
            # Aplicar el color solo en el interior del polígono (sin bordes)
            image_copy[np.where(eroded_mask == 255)] = predominant_color
    
    return image_copy

