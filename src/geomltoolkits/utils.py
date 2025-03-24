import json
import os
from typing import Any, Dict, Optional, Union

import geopandas as gpd
import mercantile
import rasterio
from rasterio.merge import merge
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
