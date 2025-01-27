import os
import random
from PIL import Image

# Configuración inicial
input_folder = r"E:\Universidad\ML\ML-Project\data\Generales-Parciales"
output_folder = os.path.join(input_folder, "random_crops")
os.makedirs(output_folder, exist_ok=True)

# Número total de subimágenes y tamaño del recorte
num_total_crops = 5600
crop_size = 200

# Obtener la lista de imágenes
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
num_images = len(image_files)
if num_images == 0:
    raise ValueError("No se encontraron imágenes en la carpeta especificada.")

# Cantidad de recortes por imagen
crops_per_image = num_total_crops // num_images

# Generar recortes aleatorios
def generate_random_crops(image_path, num_crops, crop_size, output_folder):
    with Image.open(image_path) as img:
        width, height = img.size
        for i in range(num_crops):
            if width < crop_size or height < crop_size:
                raise ValueError(f"La imagen {image_path} es más pequeña que el tamaño del recorte ({crop_size}x{crop_size}).")

            left = random.randint(0, width - crop_size)
            top = random.randint(0, height - crop_size)
            right = left + crop_size
            bottom = top + crop_size

            crop = img.crop((left, top, right, bottom))
            crop_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_crop_{i}.png"
            crop.save(os.path.join(output_folder, crop_filename))

# Procesar cada imagen
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    generate_random_crops(image_path, crops_per_image, crop_size, output_folder)

print(f"Se generaron {num_total_crops} recortes aleatorios de {crop_size}x{crop_size} y se guardaron en la carpeta {output_folder}.")
