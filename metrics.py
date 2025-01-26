import json
import csv
import math

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