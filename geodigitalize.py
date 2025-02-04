import json
from pathlib import Path
import fiona
from fiona.crs import from_epsg
from shapely.geometry import Polygon, mapping

def reconstruct_map_from_json(json_path: str, output_shapefile: str) -> None:
    """
    Reconstructs the original vectorized map from a JSON file.
    
    Each component in the JSON contains the coordinates of its vertices (in clockwise order)
    in the local coordinate system of the image, along with the information of the top left 
    corner (top_left_corner), the width, and the height of the bounding box.
    
    The transformation of each point is performed as follows:
      - new_x = top_left[0] + local_x
      - new_y = top_left[1] + (height - local_y)
    
    This corrects the vertical inversion, using the height of each component.
    
    Parameters:
      - json_path (str): Path to the input JSON file.
      - output_shapefile (str): Path to the output Shapefile.
    """
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = []

    for component, items in data.items():
        for item in items:
            local_coords = item["coordinates"]   
            top_left = item["top_left_corner"]     
            comp_height = item["height"]           
            
            transformed_coords = []
            for pt in local_coords:
                
                new_x = top_left[0] + pt[0]
                new_y = -(top_left[1] + (comp_height - pt[1]))
                transformed_coords.append((new_x, new_y))
            
            
            polygon = Polygon(transformed_coords)
            
            feature = {
                'geometry': mapping(polygon),
                'properties': {
                    'component': component,
                    'width': item.get("width", 0),
                    'height': item.get("height", 0)
                }
            }
            features.append(feature)
    
    
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'component': 'str',
            'width': 'int',
            'height': 'int'
        }
    }
    # Using EPSG:4326 (WGS84) as the reference system
    crs = from_epsg(4326)
    
    
    with fiona.open(output_shapefile, 'w', driver='ESRI Shapefile', schema=schema, crs=crs) as sink:
        for feat in features:
            sink.write(feat)
    
    print(f"Shapefile saved at: {output_shapefile}")