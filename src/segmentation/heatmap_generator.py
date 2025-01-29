import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from train import BorderDetectionCNN

def generate_heatmap(
    image_path: str,
    model_path: str = "model.pth"
) -> np.ndarray:
    """
    Generates a probability heatmap using a trained model and a sliding window approach.
    Returns the heatmap as a [0,1] array and saves the overlaid visualization with _heatmap.png suffix.

    Parameters:
    -----------
    - image_path: Path to the input image.
    - model_path: Path to the trained model checkpoint.

    Returns:
    --------
    - heatmap_normalized: np.ndarray with values [0,1], same size as the original image.
    """
    
    # Load the model and move it to GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BorderDetectionCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Read the image in color (RGB)
    original_image = Image.open(image_path).convert("RGB")
    img_array = np.array(original_image)  # Shape: (height, width, 3)
    height, width, _ = img_array.shape

    # Define window size and step size for sliding window
    WINDOW_SIZE = height // 100
    if WINDOW_SIZE < 2:
        WINDOW_SIZE = 2

    STEP_SIZE = WINDOW_SIZE // 2
    if STEP_SIZE < 1:
        STEP_SIZE = 1

    # Initialize heatmap and count map
    heatmap = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros((height, width), dtype=np.float32)

    # Input transformations (same as during training)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Slide the window over the image
    for y in range(0, height - WINDOW_SIZE, STEP_SIZE):
        for x in range(0, width - WINDOW_SIZE, STEP_SIZE):
            patch_3ch = img_array[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE, :]
            
            patch_tensor = transform(patch_3ch).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(patch_tensor)
                probs = F.softmax(logits, dim=1)
            
            prob_border = probs[0, 1].item()
            
            heatmap[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE] += prob_border
            count_map[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE] += 1

    # Normalize the heatmap to [0,1]
    count_map = np.maximum(count_map, 1e-5)
    heatmap /= count_map
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_normalized = (heatmap - h_min) / (h_max - h_min + 1e-8)

    # Generate overlay visualization and save it
    base_img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    cmap = cm.get_cmap('jet')
    heatmap_color = cmap(heatmap_normalized)[..., :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    alpha = 0.5
    overlay = cv2.addWeighted(base_img_bgr, 1.0 - alpha, heatmap_color, alpha, 0)
    

    hmp_output_dir = "heatmaps"

    base_name, _ = os.path.splitext(os.path.basename(image_path))
    output_filename = os.path.join(hmp_output_dir, f"{base_name}.png")
    cv2.imwrite(output_filename, overlay)
    print(f"Heatmap image saved to: {output_filename}")

    # Save heatmap as .hmp text file
    os.makedirs(hmp_output_dir, exist_ok=True)
    hmp_output_path = os.path.join(hmp_output_dir, f"{base_name}.hmp")
    with open(hmp_output_path, 'w') as f:
        for row in heatmap_normalized.flatten():
            f.write(f"{row} ")
    print(f"Heatmap .hmp file saved to: {hmp_output_path}")

    return heatmap_normalized

if __name__ == '__main__':
    model_path = "model.pth"
    
    input_folder = "../label_extraction/labelless_data/"
    output_heatmaps_dir = "heatmaps"
    
    os.makedirs(output_heatmaps_dir, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}")
            try:
                generate_heatmap(image_path, model_path)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
