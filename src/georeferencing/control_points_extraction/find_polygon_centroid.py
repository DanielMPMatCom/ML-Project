from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def calculate_polygon_centroid(polygon):
    """
    Calcula el centroide de un polígono en 2D usando la fórmula de
    centroides para polígonos (basado en la fórmula del "shoelace").

    :param polygon: Lista de tuplas (x, y) que representan los vértices del polígono.
                    Se asume que los puntos están en orden (horario o antihorario).
    :return: (cx, cy) tupla con las coordenadas del centroide.
    """
    # Asegurarnos de que haya al menos 3 puntos
    if len(polygon) < 3:
        raise ValueError("Se necesitan al menos 3 puntos para definir un polígono.")

    # Inicializamos área y centroides parciales
    area = 0.0
    cx = 0.0
    cy = 0.0
    
    # Recorremos los vértices. Usamos el "índice siguiente" con módulo
    for i in range(len(polygon)):
        # j es el índice del siguiente vértice, considerando el cierre del polígono
        j = (i + 1) % len(polygon)
        
        x_i, y_i = polygon[i]
        x_j, y_j = polygon[j]
        
        # Producto cruzado entre vectores
        cross = x_i * y_j - x_j * y_i
        
        area += cross
        cx += (x_i + x_j) * cross
        cy += (y_i + y_j) * cross

    # El "área" calculada por la fórmula shoelace es 2 veces el área real,
    # así que debemos dividir entre 2.
    area = area / 2.0
    
    # Si el área es cero, el polígono podría ser degenerado (línea o punto)
    if abs(area) < 1e-9:
        raise ValueError("El área del polígono es 0 (posiblemente un polígono degenerado).")
    
    # Fórmula para el centroide
    cx = cx / (6.0 * area)
    cy = cy / (6.0 * area)
    
    return cx, cy

def plot_polygon_and_centroid(vertices, centroid):
    """
    Plots the polygon and its centroid.
    
    :param vertices: List of tuples with the coordinates (x, y) of the polygon's vertices.
    :param centroid: Tuple with the coordinates (x, y) of the centroid.
    """
    polygon = Polygon(vertices)
    x, y = polygon.exterior.xy
    
    plt.figure()
    plt.plot(x, y, 'b-', label='Polygon')
    plt.plot(centroid[0], centroid[1], 'ro', label='Centroid')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Polygon and its Centroid')
    plt.legend()
    plt.show()

# Example usage with a more complex polygon
vertices = [
    (0.0, 30.0),
    (34.0, 30.0),
    (39.0, 15.0),
    (27.0, 11.0),
    (27.0, 16.0),
    (24.0, 14.0),
    (25.0, 9.0),
    (0.0, 0.0),
    (0.0, 30.0)
]
centroid = calculate_polygon_centroid(vertices)
print(f"The centroid of the polygon is: {centroid}")

plot_polygon_and_centroid(vertices, centroid) 