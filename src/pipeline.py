import os
from termcolor import colored as col
import multiprocessing
from segmentation.segmentator import segmentate_image
from vectorization.vectorize import vectorization_pipeline
from georeferencing.outliers_detection.dbscan import clustering_polygons
from termcolor import colored as col
import multiprocessing

def text_elimination(image_name):
    print(col("ELIMINATING TEXT",'blue'))
    from label_extraction import extract_labels
    extract_labels(f'./preprocessing/preprocessed_data/{image_name}.jpg',
                   './label_extraction/labelless_data/',
                   './label_extraction/detected_labels/')
   

def heatmap_generation(image_name):
    print(col("GENERATING HEATMAP",'blue'))
    from segmentation.heatmap_generator import generate_heatmap
    generate_heatmap(f'./label_extraction/labelless_data/{image_name}_labelless.jpg',
                      './segmentation/heatmaps',
                      'segmentation/model.pth')
    
def process_image(image_name):
    p1 = multiprocessing.Process(target=text_elimination,args=[image_name])
    p1.start()
    p1.join() 

    p2 = multiprocessing.Process(target=heatmap_generation,args=[image_name])
    p2.start()
    p2.join() 

    print(col('SEGMENTATING','blue'))
    segmentate_image(f'./label_extraction/labelless_data/{image_name}.png',
                      f'./segmentation/heatmaps/{image_name}.hmp',
                      f'segmentation/processed_data/{image_name}/',
                      k=25,
                      use8Way=1,
                      euclidif=1,
                      adj=1,
                      minComponentSize=400,
                      buildingBlockTreshold=0.000009
                      )
    print(col('VECTORIZING','blue'))
    vectorization_pipeline( input_dir=f'segmentation/processed_data/building_blocks/',
                             vectorize_dir=f'vectorization/vectorized_temp/',
                             components_info_path=f'segmentation/processed_data/components_info.json',
                             simplify_method="shapely",
                             simplify_tolerance=2,
                             output_directory=f'vectorization/vectorized_data/',
                             output_file_name=f'{image_name}_polygons',
                             verbose=True
                            )
    
    
    
    clustering_polygons(json_path=f'vectorization/vectorized_data/{image_name}_polygons.json',
                         output_path=f"georeferencing/outliers_detection/detected_outliers/{image_name}_outliers.json")
    
    # extract_control_points(outliers_path, control_points_path)
    
    # geodigitalize_map(vectorized_path, control_points_path, labels_path, digitalized_path)

if __name__ == '__main__':
    process_image("ohcah_cpcu_000013433")