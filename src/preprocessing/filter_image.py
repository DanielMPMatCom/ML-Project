import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

def process_map_image(image_path, output_path='processed_map.jpg'):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using histogram equalization
    enhanced_image = cv2.equalizeHist(gray_image)
    
    # Convert back to PIL for additional enhancements
    pil_image = Image.fromarray(enhanced_image)
    contrast_enhancer = ImageEnhance.Contrast(pil_image)
    final_image = contrast_enhancer.enhance(1.5)  # Increase contrast by 1.5x
    
    # Save the processed image
    final_image.save(output_path)
    
    # Display the original and processed images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(final_image, cmap='gray')
    ax[1].set_title("Processed Image")
    ax[1].axis("off")
    
    plt.show()

# Example usage (update with your actual image path)
image_path = "src/preprocessing/fana.jpg"
process_map_image(image_path)
