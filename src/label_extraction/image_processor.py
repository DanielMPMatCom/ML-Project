import os
from PIL import Image
import math
import json
import numpy as np
import sys


fillcolor = (255, 0, 255)
angles =  [360]
# angles =  [45,90,180,225,270,315,360]
# angles [45]
output_folder = "images_r"
images_pieces = {}
images_center_points = {}


def get_original_coordinates_translation(trans_x,trans_y,pixel_x,pixel_y):
    return pixel_x+trans_x,pixel_y+trans_y

import math

# def get_original_coordinates_rotation(center_x, center_y, rotated_center_x, rotated_center_y, rotated_x, rotated_y, angle):
#     # Paso 1: Aplicar la rotación inversa sin traslación
#     theta = math.radians(-angle)
#     unrotated_x = rotated_center_x + (rotated_x - rotated_center_x) * math.cos(theta) - (rotated_y - rotated_center_y) * math.sin(theta)
#     unrotated_y = rotated_center_y + (rotated_x - rotated_center_x) * math.sin(theta) + (rotated_y - rotated_center_y) * math.cos(theta)

#     # Paso 2: Corregir la traslación después de la rotación
#     original_x = unrotated_x - (rotated_center_x - center_x)
#     original_y = unrotated_y - (rotated_center_y - center_y)

#     # Debugging
#     if original_x < 0 or original_y < 0:
#         print(f'original_x: {original_x} original_y: {original_y}')
#         sys.exit()

#     return original_x, original_y


def get_original_coordinates_rotation(center_x,center_y,rotated_center_x,rotated_center_y,rotated_x,rotated_y,angle):
    trans_x = center_x - rotated_center_x
    trans_y = center_y - rotated_center_y

    if trans_x>0 or trans_y>0:
        print('transy')
        sys.exit()
    rotated_x += trans_x
    rotated_y += trans_y

    # if rotated_x<0 or rotated_y<0:
    #     print(f'rotated_x: {rotated_x} rotated_y: {rotated_y}')
    #     sys.exit()

    theta = np.radians(angle)  # Invertir el ángulo para la rotación inversa
    # Calcular las coordenadas originales
    # original_x = center_x + (rotated_x - center_x) * math.cos(theta) - (rotated_y - center_y) * math.sin(theta)
    # original_y = center_y + (rotated_x - center_x) * math.sin(theta) + (rotated_y - center_y) * math.cos(theta)

    original_x = center_x + (rotated_x - center_x) * np.cos(theta) + (rotated_y - center_y) * np.sin(theta)
    original_y = center_y - (rotated_x - center_x) * np.sin(theta) + (rotated_y - center_y) * np.cos(theta)
    # if original_x<0 or original_y<0:
    #     print(f'original_x: {original_x} original_y: {original_y}')
    #     sys.exit()
    return original_x, original_y


def rotate_and_save_image(image_path,output_folder,angles):
    rotation_paths = []
    # Crear la carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Obtener el nombre base del archivo y separarlo por el último punto
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    os.makedirs(os.path.join(output_folder,name), exist_ok=True)


    # Abrir la imagen
    with Image.open(image_path) as img:
        width, height = img.size  # Obtener las dimensiones de la imagen
        center = (width / 2, height / 2)  # Calcular el centro
        images_center_points[base_name]=center
        # Generar y guardar las imágenes rotadas
        for angle in angles:
            rotated_img = img.rotate(angle, expand=True,fillcolor=fillcolor)
            output_path = os.path.join(output_folder,name,f"{angle}")
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path,f"{name}_rotated_{angle}{ext}")
            rotated_img.save(output_path)
            rotation_paths.append(output_path)
            print(f"Imagen guardada: {output_path}")
    return rotation_paths

def split_image_with_mapping(image_path, rows, cols):

    # Abrir la imagen
    with Image.open(image_path) as img:
        width, height = img.size


        # Calcular las dimensiones de cada fragmento
        piece_width = width // cols
        piece_height = height // rows
        
        # Cortar y guardar los fragmentos
        for row in range(rows):
            for col in range(cols):
                left = col * piece_width
                top = row * piece_height
                right = left + piece_width
                bottom = top + piece_height
                
                # Recortar el fragmento
                piece = img.crop((left, top, right, bottom))
                
                parent_folder = os.path.dirname(image_path)
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                # Nombre único para el fragmento
                piece_name = f"{name}_piece_{row}_{col}{ext}"
                os.makedirs(os.path.join(parent_folder,"pieces"),exist_ok=True)
                piece_path = os.path.join(parent_folder, "pieces",piece_name)
                piece.save(piece_path)
                
                # Guardar las coordenadas del fragmento
                
                center_x, center_y = images_center_points[f"{parent_folder.split(os.sep)[-2]}{ext}"]
                images_pieces[piece_name]= (left, top,center_x,center_y,width/2,height/2,float(os.path.basename(parent_folder)))

        # Convertir el diccionario y guardar en un archivo JSON
        with open("images_info.json", "w") as images_info:
            json.dump(images_pieces, images_info)
        


def process_image(image_path,output_folder,angles,rows,cols):
    rotation_paths = rotate_and_save_image(image_path,output_folder,angles)

    for path in rotation_paths:
        split_image_with_mapping(path,rows,cols)

