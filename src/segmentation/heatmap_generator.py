import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from .train import BorderDetectionCNN

def generate_heatmap(
    image_path: str,
    model_path: str = "model.pth"
) -> np.ndarray:
    """
    Genera un heatmap de probabilidad de borde usando un modelo entrenado y un 
    enfoque de ventana deslizante. Devuelve el heatmap como matriz [0,1] y
    guarda la visualización superpuesta con sufijo '_heatmap.png'.

    Parámetros:
    -----------
    - image_path: Ruta de la imagen de entrada.
    - model_path: Ruta al modelo (checkpoint) entrenado.

    Retorna:
    --------
    - heatmap_normalized: np.ndarray con valores [0,1], mismo tamaño que la imagen original.
    """
    
    #--------------------------------------------------------------------------
    # 1. Cargar el modelo y moverlo a GPU/CPU
    #--------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BorderDetectionCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    #--------------------------------------------------------------------------
    # 2. Leer la imagen en color (RGB)
    #   * Sin convertir a escala de grises, para evitar discrepancias
    #--------------------------------------------------------------------------
    original_image = Image.open(image_path).convert("RGB")
    img_array = np.array(original_image)  # (alto, ancho, 3)
    height, width, _ = img_array.shape

    #--------------------------------------------------------------------------
    # 3. Definir tamaño y paso de la ventana deslizante
    #--------------------------------------------------------------------------
    WINDOW_SIZE = height // 75
    if WINDOW_SIZE < 2:
        WINDOW_SIZE = 2

    STEP_SIZE = WINDOW_SIZE // 2
    if STEP_SIZE < 1:
        STEP_SIZE = 1

    #--------------------------------------------------------------------------
    # 4. Inicializar heatmap y mapa de conteo
    #--------------------------------------------------------------------------
    heatmap = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    #--------------------------------------------------------------------------
    # 5. Transformaciones de entrada (mismas que en entrenamiento)
    #--------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ),
    ])

    #--------------------------------------------------------------------------
    # 6. Recorrer la imagen con la ventana deslizante
    #--------------------------------------------------------------------------
    for y in range(0, height - WINDOW_SIZE, STEP_SIZE):
        for x in range(0, width - WINDOW_SIZE, STEP_SIZE):
            # Extraer el patch
            patch_3ch = img_array[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE, :]

            # Transformar al tensor
            patch_tensor = transform(patch_3ch).unsqueeze(0).to(device)

            # Inferencia
            with torch.no_grad():
                logits = model(patch_tensor)
                probs = F.softmax(logits, dim=1)
            
            # Asumiendo que clase 0 = "no_borde" y clase 1 = "borde",
            # prob_border es la probabilidad de la clase 1
            prob_border = probs[0, 1].item()

            # Sumar la probabilidad en la región correspondiente
            heatmap[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE] += prob_border
            count_map[y : y + WINDOW_SIZE, x : x + WINDOW_SIZE] += 1

    #--------------------------------------------------------------------------
    # 7. Normalizar el heatmap [0,1]
    #--------------------------------------------------------------------------
    count_map = np.maximum(count_map, 1e-5)  # evitar división por cero
    heatmap /= count_map  # promedio de las probabilidades
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_normalized = (heatmap - h_min) / (h_max - h_min + 1e-8)

    #--------------------------------------------------------------------------
    # 8. Generar visualización superpuesta y guardarla
    #--------------------------------------------------------------------------
    # Convertir de RGB a BGR para usar cv2.addWeighted correctamente
    base_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    cmap = cm.get_cmap('jet')
    heatmap_color = cmap(heatmap_normalized)[..., :3]  # quitar canal alpha
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    alpha = 0.5  # transparencia
    overlay = cv2.addWeighted(base_img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)

    # Guardar resultado
    base_name, _ = os.path.splitext(os.path.basename(image_path))
    output_filename = f"{base_name}_heatmap.png"
    cv2.imwrite(output_filename, overlay)
    print(f"Heatmap guardado en: {output_filename}")

    return heatmap_normalized
