import math
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, rotate, translate

def scale_polygon_to_area_one(poly: Polygon) -> Polygon:
    """
    Escala un polígono (Shapely) para que su área sea 1.
    El escalado se hace respecto al centroide.
    """
    area_original = poly.area
    if area_original == 0:
        raise ValueError("El polígono tiene área 0, no se puede escalar.")
    
    # Factor de escala: s^2 * area_original = 1 => s = 1 / sqrt(area_original)
    s = 1 / math.sqrt(area_original)
    centroid = poly.centroid
    # Shapely permite escalar sobre un 'origin' (centroide en este caso)
    poly_escalado = scale(poly, xfact=s, yfact=s, origin=(centroid.x, centroid.y))
    return poly_escalado

def longest_side_index(poly: Polygon) -> int:
    """
    Retorna el índice del vértice que inicia el lado más largo de un polígono.
    Suponemos que poly.exterior.coords forma un anillo cerrado,
    es decir, el último punto es igual al primero.
    """
    coords = list(poly.exterior.coords)  # anillo cerrado
    n = len(coords) - 1  # el último repite al primero
    
    max_length = -1
    max_idx = 0
    
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        length = math.hypot(x2 - x1, y2 - y1)
        if length > max_length:
            max_length = length
            max_idx = i
    return max_idx

def align_polygon_on_side(poly: Polygon, side_index: int) -> Polygon:
    """
    Alinea el polígono de forma que:
      - El lado dado por side_index -> side_index+1 esté en el eje X.
      - El centro de ese lado coincida con el origen (0,0).
    
    Pasos:
    1. Encontrar midpoint del lado.
    2. Trasladar el polígono para que midpoint quede en (0,0).
    3. Calcular el ángulo que forma el lado con el eje X y rotar el polígono para alinearlo.
    """
    coords = list(poly.exterior.coords)
    # La lista coords está cerrada (el último punto es el mismo que el primero)
    n = len(coords) - 1
    
    (x1, y1) = coords[side_index]
    (x2, y2) = coords[(side_index + 1) % n]
    
    # Punto medio del lado
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)
    
    # Ángulo del lado respecto al eje X
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    
    # 1) Trasladar para que el midpoint sea el origen
    poly_translated = translate(poly, xoff=-mx, yoff=-my)
    
    # 2) Rotar para alinear el lado con el eje X.
    #    rotate(...) en Shapely rota en sentido antihorario, 
    #    por lo que se aplica -angle para alinear el segmento.
    poly_aligned = rotate(poly_translated, -angle, origin=(0, 0))
    
    return poly_aligned

def polygon_iou(polyA: Polygon, polyB: Polygon) -> float:
    """
    Calcula el IoU (Intersection over Union) entre dos polígonos de Shapely.
    IoU = area(intersección) / area(unión)
    """
    inter = polyA.intersection(polyB).area
    union = polyA.union(polyB).area
    if union == 0:
        return 0.0
    return inter / union

def topological_similarity(
    coordsA, 
    coordsB, 
    threshold=0.9
) -> bool:
    """
    Implementa el algoritmo propuesto para verificar si dos polígonos son
    'topológicamente iguales' por encima de un cierto umbral (threshold)
    usando el IoU máximo como criterio.
    
    Parámetros:
    - coordsA: Lista de tuplas (x, y) del polígono A en sentido clockwise.
    - coordsB: Lista de tuplas (x, y) del polígono B en sentido clockwise.
    - threshold: Valor mínimo de IoU para considerar que A y B son 'similares'.
    
    Retorna:
    - True si el IoU máximo entre A y B supera el threshold, False en caso contrario.
    """
    # 0. Encontrar polígono con menor cantidad de lados
    if len(coordsA) > len(coordsB):
        coordsA, coordsB = coordsB, coordsA

    # 1. Convertir a polígonos Shapely
    polyA = Polygon(coordsA)
    polyB = Polygon(coordsB)
    
    # 2. Escalar a área 1
    polyA = scale_polygon_to_area_one(polyA)
    polyB = scale_polygon_to_area_one(polyB)
    
    # 3. Alinear el polígono B en base a su lado más largo
    idx_longest_B = longest_side_index(polyB)
    polyB_aligned = align_polygon_on_side(polyB, idx_longest_B)
    
    # 4. Para cada lado L de A, alineamos A y calculamos IoU con el B ya alineado
    coordsA_aligned = list(polyA.exterior.coords)
    nA = len(coordsA_aligned) - 1  # polígono cerrado, último = primero
    
    max_iou = 0.0
    for i in range(nA):
        # Alinear polígono A al lado i
        polyA_aligned = align_polygon_on_side(polyA, i)
        
        # Calcular IoU con B (ya alineado)
        current_iou = polygon_iou(polyA_aligned, polyB_aligned)
        
        if current_iou > max_iou:
            max_iou = current_iou
    
    # 5. Comparar el máximo IoU encontrado con el threshold
    print(f"Máximo IoU calculado: {max_iou:.8f}")
    return max_iou >= threshold

def topological_similarity__(coordsA, coordsB, threshold=0.9) -> bool:
    """
    Verifica si dos polígonos son 'topológicamente iguales' por encima
    de un cierto umbral (threshold), usando el IoU máximo entre
    TODAS las alineaciones por lado de A y por lado de B.
    
    - coordsA, coordsB: listas de tuplas (x, y) en sentido horario (clockwise).
    - threshold: umbral del IoU para decidir si son similares.
    """
    # 1. Convertir a polígonos Shapely y escalar a área 1
    polyA = scale_polygon_to_area_one(Polygon(coordsA))
    polyB = scale_polygon_to_area_one(Polygon(coordsB))
    
    # Extraer las coordenadas (anillo cerrado) para iterar sobre los lados
    coordsA_norm = list(polyA.exterior.coords)
    coordsB_norm = list(polyB.exterior.coords)
    
    # Cantidad de lados (n-1 en lista con cierre)
    nA = len(coordsA_norm) - 1
    nB = len(coordsB_norm) - 1
    
    max_iou = 0.0
    
    # 2. Para cada lado i de A
    for i in range(nA):
        # Alinear polígono A con su lado i
        polyA_aligned = align_polygon_on_side(polyA, i)
        
        # 3. Para cada lado j de B
        for j in range(nB):
            # Alinear polígono B con su lado j
            polyB_aligned = align_polygon_on_side(polyB, j)
            
            # 4. Calcular IoU
            current_iou = polygon_iou(polyA_aligned, polyB_aligned)
            if current_iou > max_iou:
                max_iou = current_iou
    
    print(f"Máximo IoU calculado: {max_iou:.8f}")
    return max_iou >= threshold

if __name__ == "__main__":
    # EJEMPLO DE USO:
    
    # Polígono A (cuadrado de 2x2, centro en origen, área=4)
    # Coordenadas en sentido horario (clockwise)
    square_A = [(1,1.01), (1.41,0), (1,-1.007), (-1,-1), (-1,1)]
    
    # Polígono B (otro cuadrado de 4x4, mismo centro, área=16)
    # El polígono B tiene la misma forma que A, solo que más grande
    square_B = [(2,2.105), (2.027,-2), (-2.0177,-2.21), (-2,2.121)]
    
    # Debería dar True con un threshold razonable (por ejemplo 0.7)
    is_similar = topological_similarity__(square_A, square_B, threshold=0.9)
    is_similar = topological_similarity(square_A, square_B, threshold=0.9)
    print("¿Son topológicamente iguales?", is_similar)
