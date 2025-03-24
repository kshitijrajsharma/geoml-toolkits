import json
import os
from typing import Any, Dict, Optional, Union, List, Tuple

import geopandas as gpd
import mercantile
import rasterio
from rasterio.merge import merge
from shapely.geometry import mapping, shape, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.ops import unary_union, polygonize
import subprocess
import numpy as np
from PIL import Image


def merge_rasters(input_files, output_path):
    if isinstance(input_files, str):
        if os.path.isdir(input_files):
            files = []
            for root, _, fs in os.walk(input_files):
                for f in fs:
                    if f.lower().endswith(".tif"):
                        files.append(os.path.join(root, f))
            input_files = files
        else:
            raise ValueError("input_files must be a list or directory")
    elif not isinstance(input_files, list):
        raise ValueError("input_files must be a list or directory")
    src_files = [rasterio.open(fp) for fp in input_files]
    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        }
    )
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)
    for src in src_files:
        src.close()
    return output_path


def bbox2geom(bbox):
    # bbox = [float(x) for x in bbox_str.split(",")]
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]
        ],
    }
    return geometry


def load_geojson(geojson):
    """Load GeoJSON from a file path or string."""
    if isinstance(geojson, str):
        if os.path.isfile(geojson):
            with open(geojson, encoding="utf-8") as f:
                return json.load(f)
        else:
            try:
                return json.loads(geojson)
            except json.JSONDecodeError:
                raise ValueError("Invalid GeoJSON string")
    return geojson


def get_tiles(zoom, geojson=None, bbox=None, within=False):
    """
    Generate tile bounds from a GeoJSON or a bounding box.

    Args:
        geojson (str or dict): Path to GeoJSON file, GeoJSON string, or dictionary.
        bbox (tuple): Bounding box as (xmin, ymin, xmax, ymax).
        within (bool): Whether tiles must be completely within the geometry/bbox.

    Returns:
        list: List of tiles.
    """
    if geojson:
        geojson_data = load_geojson(geojson)
        bounds = generate_tiles_from_geojson(geojson_data, zoom, within)
    elif bbox:
        bounds = generate_tiles_from_bbox(bbox, zoom, within)
    else:
        raise ValueError("Either geojson or bbox must be provided.")

    return bounds


def generate_tiles_from_geojson(geojson_data, zoom, within):
    """Generate tiles based on GeoJSON data."""
    tile_bounds = []
    if geojson_data["type"] == "FeatureCollection":
        for feature in geojson_data["features"]:
            geometry = feature["geometry"]
            tile_bounds.extend(
                filter_tiles(
                    mercantile.tiles(
                        *shape(geometry).bounds, zooms=zoom, truncate=False
                    ),
                    geometry,
                    within,
                )
            )
    else:
        geometry = geojson_data
        tile_bounds.extend(
            filter_tiles(
                mercantile.tiles(*shape(geometry).bounds, zooms=zoom, truncate=False),
                geometry,
                within,
            )
        )
    return list(set(tile_bounds))


def generate_tiles_from_bbox(bbox, zoom, within):
    """Generate tiles based on a bounding box."""
    return filter_tiles(
        mercantile.tiles(*bbox, zooms=zoom, truncate=False), bbox2geom(bbox), within
    )


def filter_tiles(tiles, geometry, within=False):
    """Filter tiles to check if they are within the geometry or bbox."""
    return_tiles = []
    # print(len(list(tiles)))

    for tile in tiles:
        if within:
            if shape(mercantile.feature(tile)["geometry"]).within(shape(geometry)):
                return_tiles.append(tile)
        else:
            if shape(mercantile.feature(tile)["geometry"]).intersects(shape(geometry)):
                return_tiles.append(tile)

    return return_tiles


def load_geometry(
    input_data: Optional[Union[str, list]] = None, bbox: Optional[list] = None
) -> Optional[Dict]:
    """
    Load geometry from GeoJSON file, string, or bounding box.

    Args:
        input_data (str or list, optional): GeoJSON file path or string
        bbox (list, optional): Bounding box coordinates

    Returns:
        dict: Loaded GeoJSON geometry or None
    """
    if input_data and bbox:
        raise ValueError("Cannot provide both GeoJSON and bounding box")
    try:
        if input_data and isinstance(input_data, str):
            try:
                # Try parsing as a file
                with open(input_data, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                # If not a file, try parsing as a GeoJSON string
                return json.loads(input_data)
        elif bbox:
            # Convert bbox to GeoJSON
            xmin, ymin, xmax, ymax = bbox
            return {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax],
                            [xmin, ymin],
                        ]
                    ]
                ],
            }
        else:
            raise ValueError("Must provide either GeoJSON or bounding box")
    except Exception as e:
        raise ValueError(f"Invalid geometry input: {e}")


def get_geometry(
    geojson: Optional[Union[str, Dict]] = None, bbox: Optional[list] = None
) -> Dict[str, Any]:
    """
    Process input geometry from either a GeoJSON file/string or bounding box.

    Args:
        geojson (str or dict, optional): GeoJSON file path, string, or object
        bbox (list, optional): Bounding box coordinates [xmin, ymin, xmax, ymax]

    Returns:
        dict: Processed geometry

    Raises:
        ValueError: If both geojson and bbox are None
    """
    if geojson:
        geojson_data = load_geojson(geojson)
    elif bbox:
        geojson_data = bbox2geom(*bbox)
    else:
        raise ValueError("Supply either geojson or bbox input")

    return check_geojson_geom(geojson_data)


def check_geojson_geom(geojson: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the input GeoJSON. If it has more than one feature, perform a shapely union
    of the geometries and return the resulting geometry as GeoJSON.

    Args:
        geojson (dict): Input GeoJSON object

    Returns:
        dict: Processed GeoJSON geometry
    """
    if geojson["type"] == "FeatureCollection" and "features" in geojson:
        features = geojson["features"]
        if len(features) > 1:
            geometries = [shape(feature["geometry"]) for feature in features]
            union_geom = unary_union(geometries)
            return mapping(union_geom)
    else:
        return geojson


def split_geojson_by_tiles(
    mother_geojson_path, children_geojson_path, output_dir, prefix="OAM"
):
    # Load mother GeoJSON (osm result)
    mother_data = gpd.read_file(mother_geojson_path)

    # Load children GeoJSON (tiles)
    with open(children_geojson_path, "r", encoding="utf-8") as f:
        tiles = json.load(f)

    for tile in tiles["features"]:
        tile_geom = shape(tile["geometry"])
        tile_id = tile["properties"].get("id", tile["id"])
        x, y, z = tile_id.split("(")[1].split(")")[0].split(", ")
        x = x.split("=")[1]
        y = y.split("=")[1]
        z = z.split("=")[1]

        clipped_data = mother_data[mother_data.intersects(tile_geom)].copy()
        clipped_data = gpd.clip(clipped_data, tile_geom)

        clipped_filename = os.path.join(output_dir, f"{prefix}-{x}-{y}-{z}.geojson")
        clipped_data.to_file(clipped_filename, driver="GeoJSON", encoding="utf-8")


def detect_and_ensure_4326_geometry(geojson: Union[str, dict]) -> Dict[str, Any]:
    """
    Detect CRS from GeoJSON, convert to EPSG:4326 if needed, and union all geometries.
    
    Args:
        geojson: GeoJSON file path, string, or dictionary
        
    Returns:
        Single unioned GeoJSON geometry in EPSG:4326
    """
    if isinstance(geojson, str):
        if os.path.exists(geojson):
            gdf = gpd.read_file(geojson)
        else:
            try:
                geojson_data = json.loads(geojson)
                if "features" in geojson_data:
                    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
                else:
                    gdf = gpd.GeoDataFrame(geometry=[gpd.GeoSeries.from_json(geojson)[0]])
            except json.JSONDecodeError:
                raise ValueError(f"Invalid GeoJSON string provided")
    else:
        if "features" in geojson:
            gdf = gpd.GeoDataFrame.from_features(geojson["features"])
        else:
            gdf = gpd.GeoDataFrame(geometry=[gpd.GeoSeries.from_json(json.dumps(geojson))[0]])
    
    if gdf.crs is None:
        print("Warning: No CRS found in GeoJSON. Assuming EPSG:4326 (WGS84).")
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        original_crs = gdf.crs.to_epsg()
        print(f"Converting GeoJSON from EPSG:{original_crs} to EPSG:4326 for API compatibility...")
        gdf = gdf.to_crs(epsg=4326)
    
    unioned_geometry = unary_union(gdf.geometry.values)
    return (gpd.GeoSeries([unioned_geometry]).set_crs(epsg=4326).__geo_interface__)


def reproject_geojson(geojson_data: Dict[str, Any], target_crs: str) -> Dict[str, Any]:
    """
    Reproject GeoJSON data to the specified CRS.

    Args:
        geojson_data (dict): GeoJSON data to reproject
        target_crs (str): Target CRS (e.g., "4326" or "3857")

    Returns:
        dict: Reprojected GeoJSON data
    """
    # Skip reprojection if target is already 4326 (which is what the API returns)
    if target_crs == "4326":
        return geojson_data

    # Create a GeoDataFrame from the GeoJSON
    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    
    # Set the CRS to 4326 (the CRS of the data from the API)
    gdf.set_crs(epsg=4326, inplace=True)
    
    # Reproject to the target CRS
    gdf = gdf.to_crs(epsg=int(target_crs))
    
    # Convert back to GeoJSON
    reprojected_geojson = json.loads(gdf.to_json())
    
    # Add CRS information to the GeoJSON
    reprojected_geojson["crs"] = {
        "type": "name",
        "properties": {"name": f"urn:ogc:def:crs:EPSG::{target_crs}"}
    }
    
    return reprojected_geojson


def run_command(cmd: List[str]) -> subprocess.CompletedProcess:
    """
    Run a command via subprocess, logging its stdout and stderr.

    Args:
        cmd: Command to run as a list of strings

    Returns:
        CompletedProcess instance

    Raises:
        RuntimeError: If command fails
    """
    print("Running command: " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            print("stdout:\n" + result.stdout)
        if result.stderr:
            print("stderr:\n" + result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print("Command failed: " + " ".join(cmd))
        print(e.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def convert_tif_to_bmp(input_tif: str, output_bmp: str) -> Tuple[rasterio.Affine, str]:
    """
    Read the GeoTIFF with rasterio, then convert its first band into an 8-bit
    BMP image using Pillow. Returns the affine transform and CRS.

    Args:
        input_tif: Path to input GeoTIFF file
        output_bmp: Path to output BMP file

    Returns:
        Tuple containing the affine transform and CRS
    """
    with rasterio.open(input_tif) as src:
        # Read the first band as a NumPy array
        array = src.read(1)
        transform = src.transform
        crs = src.crs

    # Scale array to 0-255 if needed
    if array.dtype != np.uint8:
        array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(
            np.uint8
        )
    # Flip array vertically (BMP origin is at bottom-left)
    array = np.flipud(array)

    # Create a PIL image and save as BMP
    img = Image.fromarray(array)
    img.save(output_bmp, format="BMP")
    print(f"BMP image saved as {output_bmp}")
    return transform, crs


def update_geojson_coords(geojson_file: str, transform: rasterio.Affine, crs: str) -> None:
    """
    Read the GeoJSON produced by Potrace (which is in pixel coordinates),
    convert every coordinate to geographic space using the transform,
    and add CRS info.

    Args:
        geojson_file: Path to the GeoJSON file to update
        transform: Affine transform from rasterio
        crs: Coordinate Reference System as string
    """
    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)

    def convert_ring(ring):
        return [pixel_to_geo(pt, transform) for pt in ring]

    for feature in geojson_data.get("features", []):
        geom = feature.get("geometry")
        if not geom:
            continue

        if geom["type"] == "Polygon":
            new_rings = []
            for ring in geom["coordinates"]:
                new_rings.append(convert_ring(ring))
            feature["geometry"]["coordinates"] = new_rings

        elif geom["type"] == "MultiPolygon":
            new_polygons = []
            for polygon in geom["coordinates"]:
                new_polygon = []
                for ring in polygon:
                    new_polygon.append(convert_ring(ring))
                new_polygons.append(new_polygon)
            feature["geometry"]["coordinates"] = new_polygons

    # Embed the CRS (non-standard but useful)
    if crs:
        geojson_data["crs"] = {
            "type": "name",
            "properties": {"name": str(crs)},
        }

    with open(geojson_file, "w") as f:
        json.dump(geojson_data, f, indent=2)
    print(f"Updated GeoJSON saved as {geojson_file}")


def pixel_to_geo(coord: Tuple[float, float], transform: rasterio.Affine) -> List[float]:
    """
    Convert pixel coordinates (col, row) to geographic coordinates (x, y)
    using the affine transform.

    Args:
        coord: Tuple of (column, row) pixel coordinates
        transform: Affine transform from rasterio

    Returns:
        List containing [x, y] geographic coordinates
    """
    # Note: coord[0] is the column (x pixel) and coord[1] is the row (y pixel)
    x, y = transform * (coord[0], coord[1])
    return [x, y]
