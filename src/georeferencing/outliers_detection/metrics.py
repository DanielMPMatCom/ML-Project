import json
import csv
import math
from shapely.geometry import Polygon

def calculate_area(coordinates):
    n = len(coordinates)
    area = 0.5 * abs(sum(coordinates[i][0] * coordinates[(i + 1) % n][1] - coordinates[(i + 1) % n][0] * coordinates[i][1] for i in range(n)))
    return area

def calculate_perimeter(coordinates):
    n = len(coordinates)
    perimeter = sum(math.sqrt((coordinates[i][0] - coordinates[(i + 1) % n][0]) ** 2 + (coordinates[i][1] - coordinates[(i + 1) % n][1]) ** 2) for i in range(n))
    return perimeter

def calculate_bounding_box_aspect_ratio(coordinates):
    x_coords = [point[0] for point in coordinates]
    y_coords = [point[1] for point in coordinates]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    if height == 0:
        return float('inf')  # Handle division by zero
    return width / height

def calculate_compactness(area, perimeter):
    if perimeter == 0:
        return 0  # Avoid division by zero
    return area / (perimeter ** 2)

def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0  # Avoid division by zero
    return (4 * math.pi * area) / (perimeter ** 2)

def convexity_measure(coords):
    poly = Polygon(coords)
    convex_hull = poly.convex_hull
    return poly.area / convex_hull.area if convex_hull.area != 0 else 0

def calculate_polygon_sides(polygon_coords):
    # Calcular las longitudes de los lados
    side_lengths = []
    for i in range(len(polygon_coords)):
        x1, y1 = polygon_coords[i]
        x2, y2 = polygon_coords[(i + 1) % len(polygon_coords)]
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        side_lengths.append(length)

    return side_lengths

def side_length_variation(polygon_coords):
    # Calcular las longitudes de los lados
    side_lengths = calculate_polygon_sides(polygon_coords)
    
    # Calcular la media y la desviación estándar de las longitudes
    mean_length = sum(side_lengths) / len(side_lengths)
    std_dev = math.sqrt(sum((length - mean_length) ** 2 for length in side_lengths) / len(side_lengths))
    
    # Calcular el coeficiente de variación (CV)
    cv = std_dev / mean_length
    
    return cv

def side_length_variance(polygon_coords):
    #  Calcular las longitudes de los lados
    side_lengths = calculate_polygon_sides(polygon_coords)

    # Calcular la media de las longitudes
    mean_length = sum(side_lengths) / len(side_lengths)
    
    # Calcular la varianza
    variance = sum((length - mean_length) ** 2 for length in side_lengths) / len(side_lengths)
    
    return variance

