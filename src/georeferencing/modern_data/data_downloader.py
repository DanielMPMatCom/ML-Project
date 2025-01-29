import osmnx as ox
import geopandas as gpd
import shapely.ops as ops
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import json

def extraer_bloques_habana_dict(lugar = "La Habana, Cuba"):
    # 1. Definir el lugar a consultar

    # 2. Descargar la red de calles de La Habana (tipo 'drive' para calles transitables en auto)
    G = ox.graph_from_place(lugar, network_type='drive')

    # 3. Convertir los ejes (edges) a GeoDataFrame (nos quedamos solo con 'edges')
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    # 4. Crear una unión de todas las geometrías de líneas
    union_lineas = ops.unary_union(edges.geometry)

    # 5. "Poligonizar" la unión de líneas para generar polígonos (manzanas/bloques)
    bloques_poligonizados = list(ops.polygonize(union_lineas))

    # 6. Convertir los polígonos en un GeoDataFrame
    gdf_bloques = gpd.GeoDataFrame(geometry=bloques_poligonizados, crs=edges.crs)

    # Estructura que almacenará la información
    # Ejemplo: {
    #   "component_00001": [
    #       {
    #           "coordinates": [...],
    #           "top_left_corner": [...],
    #           "width": ...,
    #           "height": ...
    #       },
    #       ...
    #   ],
    #   "component_00002": [...],
    #   ...
    # }
    bloques_dict = {}

    for i, geom in enumerate(gdf_bloques.geometry, start=1):
        component_key = f"component_{i:05d}"  # Por ejemplo: "component_00001"
        bloques_dict[component_key] = []

        # Función auxiliar para procesar un polígono individual (Polygon)
        def procesa_poligono(polygon):
            # Extraer coordenadas de la frontera exterior (x, y)
            coords = list(polygon.exterior.coords)

            # Calcular bounding box (minx, miny, maxx, maxy)
            minx, miny, maxx, maxy = polygon.bounds

            # top_left_corner puede variar según tu convención de "arriba":
            # - Si consideras la típica convención cartográfica (y crece hacia arriba),
            #   podrías usar [minx, maxy] como "esquina superior izquierda".
            # - Si consideras la típica convención de imágenes (origen arriba-izquierda),
            #   podrías usar [minx, miny] y asumir que "top" es el valor menor de y.
            # En tu ejemplo, parece que se usa [minx, miny].
            top_left_corner = [minx, miny]

            width = maxx - minx
            height = maxy - miny

            return {
                "coordinates": [[float(x), float(y)] for x, y in coords],
                "top_left_corner": top_left_corner,
                "width": float(width),
                "height": float(height)
            }

        # Verificar si la geometría es Polygon o MultiPolygon
        if isinstance(geom, Polygon):
            # Procesar directamente el polígono
            bloque_info = procesa_poligono(geom)
            bloques_dict[component_key].append(bloque_info)

        elif isinstance(geom, MultiPolygon):
            # Si es MultiPolygon, procesar cada sub-polígono
            for subpoly in geom.geoms:
                bloque_info = procesa_poligono(subpoly)
                bloques_dict[component_key].append(bloque_info)

    # 7. Visualizar los polígonos resultantes (opcional)
    ax = gdf_bloques.plot(figsize=(10, 10), color="lightblue", edgecolor="black")
    ax.set_title("Bloques (manzanas) de La Habana", fontsize=14)
    ax.set_xlim(-82.45, -82.35)
    ax.set_ylim(23.10, 23.15)
    plt.show()

    return bloques_dict

if __name__ == "__main__":
    datos_bloques = extraer_bloques_habana_dict()
    
    # Si quieres mostrarlo por pantalla con formato JSON:
    print(json.dumps(datos_bloques, indent=4, ensure_ascii=False))
    
    # O guardarlo en un archivo JSON:
    with open("modern_data.json", "w", encoding="utf-8") as f:
        json.dump(datos_bloques, f, indent=4, ensure_ascii=False)
