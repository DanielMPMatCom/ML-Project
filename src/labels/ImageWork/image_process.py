import os
from PIL import Image
import math
import json
import numpy as np


fillcolor = (255, 0, 255)
angles =  [45, 90, 135, 180, 225, 270, 315, 360]
output_folder = "images_r"
images_pieces = {}
images_center_points = {}



def get_image_center(image_path):
    with Image.open(image_path) as img:
        width, height = img.size  # Obtener las dimensiones de la imagen
        center = (width / 2, height / 2)  # Calcular el centro
        return center


# def get_original_coordinates_rotation(center_x, center_y,rotated_x, rotated_y, angle):
#     # Convertir el ángulo a radianes
#     theta = math.radians(-angle)  # Invertir el ángulo para la rotación inversa
    
#     # Calcular las coordenadas originales
#     original_x = center_x + (rotated_x - center_x) * math.cos(theta) - (rotated_y - center_y) * math.sin(theta)
#     original_y = center_y + (rotated_x - center_x) * math.sin(theta) + (rotated_y - center_y) * math.cos(theta)
#     return original_x, original_y

def get_original_coordinates_translation(trans_x,trans_y,pixel_x,pixel_y):
    return pixel_x+trans_x,pixel_y+trans_y



def get_original_coordinates_rotation(center_x, center_y,x_rotado,y_rotado, angle):
    # Convertir el ángulo a radianes (negativo para invertir la rotación)
    angulo = np.radians(-angle)
 

    # Calcular el tamaño expandido
    diagonal = np.sqrt((center_x*2)**2 + (center_y*2)**2)
    ancho_expandido = int(diagonal * abs(np.cos(angulo)) + diagonal * abs(np.sin(angulo)))
    alto_expandido = ancho_expandido  # El mismo para ambos por expansión máxima

    # Nuevo centro tras la expansión
    cx_expandido, cy_expandido = ancho_expandido / 2, alto_expandido / 2

    # Trasladar las coordenadas del punto rotado al centro expandido
    x_trasladado = x_rotado - cx_expandido
    y_trasladado = y_rotado - cy_expandido

    # Matriz de rotación inversa
    rotacion_inversa = np.array([
        [np.cos(angulo), np.sin(angulo)],
        [-np.sin(angulo), np.cos(angulo)]
    ])

    # Aplicar la rotación inversa
    x_original, y_original = np.dot(rotacion_inversa, [x_trasladado, y_trasladado])

    # Regresar al sistema de coordenadas original
    x_original += center_x
    y_original += center_y

    return x_original, y_original





def rotate_and_save_image(image_path,output_folder):
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
                images_pieces[piece_name]= (left, top,center_x,center_y,float(os.path.basename(parent_folder)))

        # Convertir el diccionario y guardar en un archivo JSON
        with open("images_info.json", "w") as images_info:
            json.dump(images_pieces, images_info)
        


def process_image(image_path):
    rotation_paths = rotate_and_save_image(image_path,output_folder)

    for path in rotation_paths:
        split_image_with_mapping(path,4,4)


    
# image_path = "images/ohcah_cpcu_000013481.jpg"
# # rotate_and_save_image(image_path,output_folder)

# # split_image_with_mapping("images_r/ohcah_cpcu_000013481/45/ohcah_cpcu_000013481_rotated_45.jpg",output_folder,4,4)

# process_image(image_path)

# for clave, valor in images_pieces.items():
#     print(f"{clave}: {valor}")