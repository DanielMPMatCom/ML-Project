import json
from shapely.geometry import Polygon
from find_polygon_centroid import calculate_polygon_centroid
from polygons_comparation import topological_similarity

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def match_outliers(old_outliers_path, new_outliers_path, old_polygons_path, new_polygons_path, output_path):
    old_outliers = load_json(old_outliers_path)
    new_outliers = load_json(new_outliers_path)
    old_polygons = load_json(old_polygons_path)
    new_polygons = load_json(new_polygons_path)
    
    control_points = []
    
    for old_id in old_outliers:
        old_coords = old_polygons[old_id][0]['coordinates']
        top_left_corner = old_polygons[old_id][0]["top_left_corner"]
        
        for new_id in new_outliers:
            new_coords = new_polygons[new_id][0]['coordinates']
            
            if topological_similarity(old_coords, new_coords):
                centroid_x, centroid_y = calculate_polygon_centroid(old_coords)
                pixel_x = top_left_corner[0] + centroid_x
                pixel_y = top_left_corner[1] + centroid_y
                latitude, longitude = calculate_polygon_centroid(new_coords)
                control_points.append([pixel_x, pixel_y, latitude, longitude])
    
    save_json(control_points, output_path)
    print(f"Matched outliers saved to {output_path}")
