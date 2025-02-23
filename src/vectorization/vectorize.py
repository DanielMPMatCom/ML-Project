import os
import cv2
import json
import fiona
import subprocess
from rdp import rdp
from osgeo import gdal
from pathlib import Path
from shapely.affinity import scale
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import sys

# region Main Pipeline


def vectorization_pipeline(
    input_dir: str,
    vectorize_dir: str,
    components_info_path: str,
    simplify_method: str,
    simplify_tolerance: float,
    output_directory: str,
    output_file_name: str,
    verbose: bool = False,
) -> None:
    """
    Executes a vectorization pipeline that processes input files, simplifies shapes,
    and groups shapefile data into a JSON output.
    Parameters:
    - input_dir (str): Directory containing the input files to be processed.
    - vectorize_dir (str): Directory where intermediate vectorized files will be stored.
    - components_info_path (str): Path to the file containing components information.
    - simplify_method (str): Method to use for simplifying shapes. Must be either 'rdp' or 'shapely'.
    - simplify_tolerance (float): Tolerance level for shape simplification.
    - output_directory (str): Directory where the final output JSON file will be stored.
    - output_file_name (str): Name of the output JSON file (without extension).
    - verbose (bool, optional): If True, enables verbose logging. Default is False.
    Raises:
    - ValueError: If an invalid simplify_method is provided.
    """

    os.makedirs(vectorize_dir, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)

    simplify_method = simplify_method.lower()
    if simplify_method not in ["rdp", "shapely"]:
        raise ValueError("Invalid simplify_method. Choose either 'rdp' or 'shapely'.")

    process_directory(input_dir=input_dir, output_dir=vectorize_dir, verbose=verbose)

    process_shapefiles(
        input_directory=vectorize_dir,
        tolerance=simplify_tolerance,
        simplify_method=simplify_method,
        verbose=verbose,
    )

    input_dir_for_group = vectorize_dir + "/" + simplify_method
    output_json_path = os.path.join(output_directory, f"{output_file_name}.json")

    group_shapefile_data(
        input_directory=input_dir_for_group,
        components_info_path=components_info_path,
        output_json_path=output_json_path,
        verbose=verbose,
    )


# region Vectorization


def preprocess_image(input_path: str, output_path: str) -> None:
    """
    Preprocess the image: convert to grayscale and binarize.
    Save the result as a GeoTIFF file.
    """

    # Read the image
    img = cv2.imread(input_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert the image so the background is black and the polygon is white
    inverted = cv2.bitwise_not(gray)

    # Apply binarization (using a fixed threshold)
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the binarized image as GeoTIFF
    driver = gdal.GetDriverByName("GTiff")
    out_dataset = driver.Create(
        output_path, binary.shape[1], binary.shape[0], 1, gdal.GDT_Byte
    )
    out_dataset.GetRasterBand(1).WriteArray(binary)
    out_dataset.SetProjection("EPSG:4326")  # WGS84 projection


def vectorize_image(input_raster, output_vector) -> None:
    """
    Uses GDAL to vectorize the binarized image and generate a Shapefile.
    """
    # Get the directory of the 'osgeo' package
    osgeo_dir = os.path.dirname(gdal.__file__)
    # Construct the path to the gdal_polygonize.py script located in osgeo_utils
    gdal_polygonize_path = os.path.abspath(os.path.join(osgeo_dir, "..", "osgeo_utils", "gdal_polygonize.py"))
    
    # Print the computed path for debugging (optional)
    # print("Using gdal_polygonize script at:", gdal_polygonize_path)
    
    # Run the script with the current Python interpreter
    subprocess.run(
        [sys.executable, gdal_polygonize_path, input_raster, "-f", "ESRI Shapefile", output_vector],
        check=True
    )


def process_directory(input_dir: str, output_dir: str, verbose: bool = False) -> None:
    """
    Processes all JPG images in a directory.
    """
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_dir, filename)
            output_tiff_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.tif"
            )
            output_shapefile_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.shp"
            )

            preprocess_image(input_path, output_tiff_path)

            vectorize_image(output_tiff_path, output_shapefile_path)
            if verbose:
                print(f"Processed {filename}, result saved in {output_shapefile_path}")


# region Polygon Filtering


def filter_polygons(input_shapefile: str, output_shapefile: str) -> None:
    """
    Filters the main polygon (black in the binarized image) based on its area or position.
    """
    with fiona.open(input_shapefile, "r") as source:
        schema = source.schema
        crs = source.crs
        polygons = []

        for feature in source:
            geom = shape(feature["geometry"])
            if geom.geom_type == "Polygon" or geom.geom_type == "MultiPolygon":
                polygons.append(geom)

        # Filter the largest polygon (you can adjust this logic if necessary)
        main_polygon = max(polygons, key=lambda x: x.area)

        # Create shapefile with only the polygon of interest
        with fiona.open(
            output_shapefile, "w", driver=source.driver, schema=schema, crs=crs
        ) as sink:
            feature = {
                "type": "Feature",
                "geometry": mapping(main_polygon),
                "properties": {
                    key: None for key in schema["properties"]
                },  # Clear original properties
            }
            sink.write(feature)


# region Polygon Simplification


def simplify_shapefile_with_rdp(
    input_path: str, output_path: str, tolerance: float
) -> None:
    """
    Simplifies a shapefile using the Ramer-Douglas-Peucker (RDP) algorithm.
    """
    with fiona.open(input_path, "r") as source:
        schema = source.schema
        crs = source.crs

        # Create output file
        with fiona.open(
            output_path, "w", driver=source.driver, schema=schema, crs=crs
        ) as sink:
            for feature in source:
                geom = shape(feature["geometry"])
                if geom.geom_type == "Polygon":
                    simplified_coords = rdp(
                        list(geom.exterior.coords), epsilon=tolerance
                    )
                    simplified_geom = Polygon(simplified_coords)
                elif geom.geom_type == "MultiPolygon":
                    simplified_geom = MultiPolygon(
                        [
                            Polygon(rdp(list(poly.exterior.coords), epsilon=tolerance))
                            for poly in geom.geoms
                        ]
                    )
                else:
                    continue  # Ignore unsupported geometries
                feature["geometry"] = fiona.Geometry.from_dict(mapping(simplified_geom))
                sink.write(feature)


def simplify_shapefile_with_shapely(
    input_path: str, output_path: str, tolerance: float
) -> None:
    """
    Simplifies a shapefile using shapely.simplify.
    """
    with fiona.open(input_path, "r") as source:
        schema = source.schema
        crs = source.crs

        # Create output file
        with fiona.open(
            output_path, "w", driver=source.driver, schema=schema, crs=crs
        ) as sink:
            for feature in source:
                geom = shape(feature["geometry"])
                if geom.geom_type in ["Polygon", "MultiPolygon"]:
                    simplified_geom = geom.simplify(tolerance, preserve_topology=True)
                else:
                    continue  # Ignore unsupported geometries
                feature["geometry"] = fiona.Geometry.from_dict(mapping(simplified_geom))
                sink.write(feature)


# region Flip Vertically


def flip_vertical(input_path: str, output_path: str) -> None:
    """
    Flips the polygons in a shapefile vertically.
    """
    with fiona.open(input_path, "r") as source:
        schema = source.schema
        crs = source.crs

        # Create output file
        with fiona.open(
            output_path, "w", driver=source.driver, schema=schema, crs=crs
        ) as sink:
            for feature in source:
                geom = shape(feature["geometry"])
                flipped_geom = scale(geom, xfact=1, yfact=-1, origin="center")
                feature["geometry"] = fiona.Geometry.from_dict(mapping(flipped_geom))
                sink.write(feature)


def process_shapefiles(
    input_directory: str,
    tolerance: float,
    simplify_method: str,
    verbose=False,
) -> None:
    """
    Processes all shapefiles in a directory: filters the polygon of interest, applies simplifications, and flips vertically.
    """

    input_path = Path(input_directory)
    filtered_output_dir = input_path / "filtered"
    output_dir = input_path / simplify_method

    filtered_output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for shp_file in input_path.glob("*.shp"):
        filtered_output_path = filtered_output_dir / shp_file.name
        output_path = output_dir / shp_file.name

        if verbose:
            print(f"Filtering the main polygon of {shp_file.name}...")
        filter_polygons(str(shp_file), str(filtered_output_path))

        if simplify_method == "rdp":
            if verbose:
                print(f"Simplifying {shp_file.name} with RDP...")
            simplify_shapefile_with_rdp(
                str(filtered_output_path), str(output_path), tolerance
            )

        if simplify_method == "shapely":
            if verbose:
                print(f"Simplifying {shp_file.name} with Shapely...")
            simplify_shapefile_with_shapely(
                str(filtered_output_path), str(output_path), tolerance
            )

        if verbose:
            print(f"Flipping {shp_file.name} vertically...")
        flip_vertical(str(output_path), str(output_path))

        if verbose:
            print(f"Processed {shp_file.name}: saved in {output_path}")


# region Save data


def extract_polygon_data(shapefile_path: str) -> list:
    """
    Extracts the vertex coordinates of polygons in clockwise order from a shapefile.
    """
    polygons_data = []

    with fiona.open(shapefile_path, "r") as shapefile:
        for feature in shapefile:
            geom = shape(feature["geometry"])
            if geom.geom_type == "Polygon":
                # Ensure that the coordinates are in clockwise order
                if not geom.exterior.is_ccw:
                    coords = list(geom.exterior.coords)
                else:
                    coords = list(Polygon(geom.exterior.coords[::-1]).exterior.coords)
                polygons_data.append({"coordinates": coords})
            elif geom.geom_type == "MultiPolygon":
                for polygon in geom.geoms:
                    # Ensure that the coordinates are in clockwise order
                    if not polygon.exterior.is_ccw:
                        coords = list(polygon.exterior.coords)
                    else:
                        coords = list(
                            Polygon(polygon.exterior.coords[::-1]).exterior.coords
                        )
                    polygons_data.append({"coordinates": coords})

    return polygons_data


def parse_components_info(components_info_path: str) -> dict:
    """
    Parses the components_info.json file to extract data for each component.
    """
    components_data = {}

    with open(components_info_path, "r") as file:
        data = json.load(file)

        for component in data:
            component_id = str(component["component"])
            components_data[component_id] = {
                "top_left_corner": (
                    component["topLeftCorner"]["x"],
                    component["topLeftCorner"]["y"],
                ),
                "width": component["width"],
                "height": component["height"],
            }

    return components_data


def group_shapefile_data(
    input_directory: str,
    components_info_path: str,
    output_json_path: str,
    verbose=False,
) -> None:
    """
    Groups shapefile data with the same base name, adds data from the components_info.txt file,
    and saves the information in a JSON file.
    """
    input_dir = Path(input_directory)
    shapefiles = list(input_dir.glob("*.shp"))
    grouped_data = {}

    # Read data from the components_info.json file
    components_data = parse_components_info(components_info_path)

    # Iterate over all .shp files in the directory
    for shapefile_path in shapefiles:
        base_name = shapefile_path.stem

        # Extract vertex coordinates
        vertices_data = extract_polygon_data(str(shapefile_path))

        # Add data to the corresponding group
        if base_name not in grouped_data:
            grouped_data[base_name] = []
        grouped_data[base_name].extend(vertices_data)

        # If the shapefile has a name like "component_0000x", add top_left_corner, width, and height
        component_number = base_name.split("_")[-1].lstrip(
            "0"
        )  # Extract the component number
        if component_number in components_data:
            for item in grouped_data[base_name]:
                item["top_left_corner"] = components_data[component_number][
                    "top_left_corner"
                ]
                item["width"] = components_data[component_number].get("width")
                item["height"] = components_data[component_number].get("height")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Save the grouped data to a JSON file
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(grouped_data, json_file, ensure_ascii=False, indent=4)

    if verbose:
        print(f"JSON file saved at: {output_json_path}")
