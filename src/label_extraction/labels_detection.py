from .image_processor import get_original_coordinates_rotation,get_original_coordinates_translation


from argparse import Namespace
from .PaddleOCR import paddleocr
import os
import json
import numpy as np
from sklearn.cluster import KMeans

original_labels = []

def detect_image_labels(image_dir,image_output):
    # Configuración de argumentos
    args = Namespace(
        use_gpu=False,
        use_xpu=False,
        use_npu=False,
        use_mlu=False,
        use_gcu=False,
        ir_optim=True,
        use_tensorrt=False,
        min_subgraph_size=15,
        precision="fp32",
        gpu_mem=500,
        gpu_id=0,
        image_dir=image_dir,
        page_num=0,
        det_algorithm="DB",
        det_model_dir="./label_extraction/PaddleOCR/inference/det/en_PP-OCRv3_det_infer/",
        det_limit_side_len=960,
        det_limit_type="max",
        det_box_type="quad",
        det_db_thresh=0.3,
        det_db_box_thresh=0.6,
        det_db_unclip_ratio=1.5,
        enable_mkldnn = False,
        max_batch_size=10,
        use_dilation=False,
        det_db_score_mode="fast",
        rec_algorithm="SVTR_LCNet",
        rec_model_dir="./label_extraction/PaddleOCR/inference/reg/en_PP-OCRv3_rec_infer/",
        rec_image_inverse=True,
        rec_image_shape="3, 48, 320",
        rec_batch_num=6,
        max_text_length=25,
        rec_char_dict_path="./label_extraction/PaddleOCR/ppocr/utils/en_dict.txt",
        use_space_char=True,
        vis_font_path="./label_extraction/PaddleOCR/doc/fonts/simfang.ttf",
        drop_score=0.5,
        draw_img_save_dir=image_output,
        save_crop_res=False,
        # crop_res_save_dir="./output",
        crop_res_save_dir="",
        use_mp=False,
        use_angle_cls=False,
        total_process_num=1,
        process_id=0,
        benchmark=False,
        # save_log_path="./log_output/",
        save_log_path="",
        show_log=True,
        use_onnx=False,
        onnx_providers=False,
        onnx_sess_options=False,
        return_word_box=False,
        warmup=False
    )

    # Llamar a la función principal
    return paddleocr.predict_system.detect_labels(args)

def process_directories(base_directory,labels_detected_dir):
    original_labels = []
    count = 0
    flag = False
    """
    Recorre todas las carpetas del directorio base, entra en la carpeta 'pieces' y procesa los archivos.
    """
    with open("images_info.json", "r") as images_info:
        images_pieces = json.load(images_info)
        print('images_pieces',images_pieces)
    base_name = os.path.basename(base_directory)
    for root, dirs, _ in os.walk(base_directory):
        pieces_output = os.path.join(root, labels_detected_dir)
        if os.path.basename(root) == base_name:
            print('root',root)
            os.makedirs(pieces_output, exist_ok=True)
        if 'pieces' in dirs:  # Busca la carpeta 'pieces'
            pieces_path = os.path.join(root, 'pieces')
            for file_name in os.listdir(pieces_path):  # Itera por cada archivo en 'pieces'
                image_path = os.path.join(pieces_path, file_name)
                trans_x,trans_y,center_x,center_y,rotated_center_x,rotated_center_y,angle = images_pieces[file_name]
                
                piece_labels = detect_image_labels(image_path,pieces_output)
                count += len(piece_labels)
                for pol in piece_labels:
                    original_pol = np.zeros((4, 2), dtype=np.float32)
                    for i, (pixel_x, pixel_y) in enumerate (pol):
                        original_coordinates = get_original_coordinates_translation(trans_x,trans_y,pixel_x,pixel_y)
                        # if rotated_x<0 or rotated_y<0:
                        #     flag = True
                        #     print(f'trans {rotated_x} - {rotated_y}')
                        #original_coordinates = get_original_coordinates_rotation(center_x,center_y,rotated_center_x,rotated_center_y,rotated_x,rotated_y,angle)
                        if original_coordinates[0]<0 or original_coordinates[1]<0:
                            flag = True
                            # print(f'rotated {original_coordinates[0]} - {original_coordinates[1]}')
                        original_pol[i]=original_coordinates
                    original_labels.append(original_pol)
    return original_labels
    # print("--------------------------------------------")
    # print(f'flag: {flag}')    



