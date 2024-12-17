import os
import cv2
import numpy as np
import gc

base_path = r"./data"
new_base_path = r"./new_data"

if not os.path.exists(new_base_path):
    os.makedirs(new_base_path)

# Kernel para operaciones morfológicas
kernel = np.ones((3,3), np.uint8)

for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.lower().endswith('.jpg'):
            original_path = os.path.join(root, file)
            rel_path = os.path.relpath(original_path, base_path)
            new_path = os.path.join(new_base_path, rel_path)

            new_dir = os.path.dirname(new_path)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            img = cv2.imread(original_path)
            if img is None:
                continue

            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Suavizar para reducir ruido (grano)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)

            # Aplicar umbral Otsu
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            # Verificar si debemos invertir (queremos fondo blanco, líneas negras)
            white_pixels = np.sum(binary == 255)
            black_pixels = np.sum(binary == 0)
            if black_pixels > white_pixels:
                binary = cv2.bitwise_not(binary)

            # Operaciones morfológicas para limpiar ruido y mejorar líneas
            # Opening: elimina ruido blanco pequeño
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            # Closing: cierra pequeños agujeros en las líneas negras
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            cv2.imwrite(new_path, binary)

            del img, gray, blurred, binary
            gc.collect()
