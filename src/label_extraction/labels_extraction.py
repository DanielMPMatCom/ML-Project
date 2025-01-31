from .image_processor import process_image
from .labels_detection import process_directories
from .labels_removal import remove_text_with_surrounding_color
from .PaddleOCR import paddleocr

import os
import numpy as np
import cv2
from PIL import Image
import math
import shutil


# Global variables
images_chunks_dir = "images_chunks"
pieces_labels_detected_dir = "labels_detected"
angles = [360]
rows = 8
cols = 8

# Main function
def extract_labels(image_path,labelless_path,colored_labels_path,detected_labels_path=''):
    
    image_name = os.path.basename(image_path)
    name, ext = os.path.splitext(image_name)

    img = cv2.imread(image_path)
    width, height = img.shape[:2]

    folder_path = os.path.join(images_chunks_dir,name) 
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    process_image(image_path=image_path,output_folder=images_chunks_dir,angles=angles,rows = max(1,math.ceil(width / 1000)),cols = max(1,math.ceil(height / 1000)))
    original_labels = process_directories(os.path.join(images_chunks_dir,name),pieces_labels_detected_dir)
    original_labels = paddleocr.predict_system.sorted_boxes(np.array(original_labels))
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw_img = paddleocr.predict_system.draw_ocr_box_txt(
    image=image,
    boxes=original_labels,
    drop_score=0.5,
    font_path="./PaddleOCR/doc/fonts/simfang.ttf"
    )

    cv2.imwrite(
    os.path.join(colored_labels_path, f'{name}_colored_labels{ext}'),
    draw_img[:, :, ::-1],
    )
    result = remove_text_with_surrounding_color(image_path, np.array(original_labels).astype(np.int32))
    cv2.imwrite(os.path.join(labelless_path,f'{name}_labelless{ext}'), result)


