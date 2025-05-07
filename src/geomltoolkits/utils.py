import json
import os
from typing import Any, Dict, Optional, Union

import geopandas as gpd
import mercantile
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


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

def georeference_tile(
    input_tiff: str,
    x: int,
    y: int,
    z: int,
    output_tiff: str,
    crs: str = "4326",
    overlap_pixels: int = 0
) -> str:
    """
    Georeference a TIFF image based on tile coordinates (x, y, z) with optional overlap.
    
    Args:
        input_tiff: Path to input TIFF file
        x: Tile x coordinate
        y: Tile y coordinate
        z: Tile z coordinate (zoom level)
        output_tiff: Path to save the georeferenced output file
        crs: Coordinate reference system (4326 or 3857)
        overlap_pixels: Number of pixels to expand the tile bounds by
        
    Returns:
        Path to georeferenced TIFF file
    """
    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_tiff)), exist_ok=True)
    
    with rasterio.open(input_tiff) as src:
        kwargs = src.meta.copy()
        
        if overlap_pixels > 0:
            if crs == "3857":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                xmin, ymin = transformer.transform(bounds.west, bounds.south)
                xmax, ymax = transformer.transform(bounds.east, bounds.north)
                
                x_res = (xmax - xmin) / 256
                y_res = (ymax - ymin) / 256
                
                xmin -= (overlap_pixels * x_res)
                ymin -= (overlap_pixels * y_res)
                xmax += (overlap_pixels * x_res)
                ymax += (overlap_pixels * y_res)

                mercator_bounds = (xmin, ymin, xmax, ymax)
                transform = from_bounds(*mercator_bounds, 256, 256)
            else:
                x_res = (bounds.east - bounds.west) / 256
                y_res = (bounds.north - bounds.south) / 256
                
                adjusted_bounds = (
                    bounds.west - (overlap_pixels * x_res),
                    bounds.south - (overlap_pixels * y_res),
                    bounds.east + (overlap_pixels * x_res),
                    bounds.north + (overlap_pixels * y_res)
                )
                
                transform = from_bounds(*adjusted_bounds, 256, 256)
        else:
            if crs == "3857":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                xmin, ymin = transformer.transform(bounds.west, bounds.south)
                xmax, ymax = transformer.transform(bounds.east, bounds.north)
                mercator_bounds = (xmin, ymin, xmax, ymax)
                transform = from_bounds(*mercator_bounds, 256, 256)
            else:
                transform = from_bounds(*bounds, 256, 256)
        
        kwargs.update({
            'crs': rasterio.CRS.from_epsg(3857 if crs == "3857" else 4326),
            'transform': transform
        })
        
        with rasterio.open(output_tiff, 'w', **kwargs) as dst:
            dst.write(src.read())
            dst.update_tags(ns="rio_georeference", georeferencing_applied="True")
            if overlap_pixels > 0:
                dst.update_tags(ns="rio_georeference", overlap_applied=str(overlap_pixels))
    
    # print(f"Successfully georeferenced tile to EPSG:{crs} with {overlap_pixels} pixels overlap: {output_tiff}")
    return output_tiff


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
                        *shape(geometry).bounds, zooms=zoom, truncate=True
                    ),
                    geometry,
                    within,
                )
            )
    else:
        geometry = geojson_data
        tile_bounds.extend(
            filter_tiles(
                mercantile.tiles(*shape(geometry).bounds, zooms=zoom, truncate=True),
                geometry,
                within,
            )
        )
    return list(set(tile_bounds))


def generate_tiles_from_bbox(bbox, zoom, within):
    """Generate tiles based on a bounding box."""
    return filter_tiles(
        mercantile.tiles(*bbox, zooms=zoom, truncate=True), bbox2geom(bbox), within
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


import glob
import re


def georeference_prediction_tiles(
    prediction_path: str,
    georeference_path: str,
    overlap_pixels: int = 0,
) :
    """
    Georeference all prediction tiles based on their embedded x,y,z coordinates in filenames.
    
    Args:
        prediction_path: Directory containing prediction tiles
        georeference_path: Directory to save georeferenced tiles
        tile_overlap_distance: Overlap distance between tiles
        
    Returns:
        List of paths to georeferenced tiles
    """
    print('test senorita')
    os.makedirs(georeference_path, exist_ok=True)
    
    image_files = glob.glob(os.path.join(prediction_path, "*.png"))
    image_files.extend(glob.glob(os.path.join(prediction_path, "*.jpeg")))
    
    georeferenced_files = []
    
    for image_file in image_files:
        filename = os.path.basename(image_file)
        filename_without_ext = re.sub(r"\.(png|jpeg)$", "", filename)
        
        try:
            parts = re.split("-", filename_without_ext)
            if len(parts) >= 3:
                # Get the last three parts which should be x, y, z
                x_tile, y_tile, zoom = map(int, parts[-3:])
                
                output_tiff = os.path.join(georeference_path, f"{filename_without_ext}.tif")
                
                georeferenced_file = georeference_tile(
                    input_tiff=image_file,
                    x=x_tile,
                    y=y_tile,
                    z=zoom,
                    output_tiff=output_tiff,
                    crs="4326",
                    overlap_pixels=overlap_pixels
                )
                
                georeferenced_files.append(georeferenced_file)
                # print(f"Georeferenced {filename} to {output_tiff}")
            else:
                print(f"Warning: Could not extract tile coordinates from {filename}")
                
        except Exception as e:
            print(f"Error georeferencing {filename}: {str(e)}")
    
    print(f"Georeferenced {len(georeferenced_files)} tiles to {georeference_path}")
    return georeference_path