import json
import os
from typing import Any

import geopandas as gpd
import mercantile
import numpy as np
import rasterio.features as rio_features
from rasterio.transform import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from .._logging import get_logger

log = get_logger(__name__)


def bbox2geom(bbox: list[float]) -> dict[str, Any]:
    """Convert a bounding box [xmin, ymin, xmax, ymax] to a GeoJSON Polygon geometry."""
    return {
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


def load_geojson(geojson: str | dict) -> dict:
    """Load GeoJSON from a file path, JSON string, or dict passthrough."""
    if isinstance(geojson, str):
        if os.path.isfile(geojson):
            with open(geojson, encoding="utf-8") as f:
                return json.load(f)
        try:
            return json.loads(geojson)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid GeoJSON string") from exc
    return geojson


def load_geometry(
    input_data: str | None = None,
    bbox: list[float] | None = None,
) -> dict:
    """Load geometry from GeoJSON file, string, or bounding box."""
    if input_data and bbox:
        raise ValueError("Cannot provide both GeoJSON and bounding box")
    if input_data and isinstance(input_data, str):
        try:
            with open(input_data, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return json.loads(input_data)
    elif bbox:
        xmin, ymin, xmax, ymax = bbox
        return bbox2geom([xmin, ymin, xmax, ymax])
    raise ValueError("Must provide either GeoJSON or bounding box")


def check_geojson_geom(geojson: dict[str, Any]) -> dict[str, Any]:
    """Union multiple features in a FeatureCollection into a single geometry."""
    if geojson.get("type") == "FeatureCollection" and "features" in geojson:
        features = geojson["features"]
        if len(features) > 1:
            geometries = [shape(feature["geometry"]) for feature in features]
            return mapping(unary_union(geometries))
        if len(features) == 1:
            return features[0]["geometry"]
        raise ValueError("FeatureCollection has no features")
    return geojson


def get_tiles(
    zoom: int,
    geojson: str | dict | None = None,
    bbox: tuple | list | None = None,
    within: bool = False,
) -> list:
    """Generate tile bounds from a GeoJSON or a bounding box at a given zoom level."""
    if geojson:
        geojson_data = load_geojson(geojson)
        return _tiles_from_geojson(geojson_data, zoom, within)
    if bbox:
        return _tiles_from_bbox(bbox, zoom, within)
    raise ValueError("Either geojson or bbox must be provided.")


def _tiles_from_geojson(geojson_data: dict, zoom: int, within: bool) -> list:
    tile_bounds = []
    if geojson_data["type"] == "FeatureCollection":
        for feature in geojson_data["features"]:
            geometry = feature["geometry"]
            raw_tiles = mercantile.tiles(*shape(geometry).bounds, zooms=zoom, truncate=True)
            tile_bounds.extend(filter_tiles(raw_tiles, geometry, within))
    else:
        geometry = geojson_data
        raw_tiles = mercantile.tiles(*shape(geometry).bounds, zooms=zoom, truncate=True)
        tile_bounds.extend(filter_tiles(raw_tiles, geometry, within))
    return list(set(tile_bounds))


def _tiles_from_bbox(bbox: tuple | list, zoom: int, within: bool) -> list:
    return filter_tiles(
        mercantile.tiles(*bbox, zooms=zoom, truncate=True),
        bbox2geom(list(bbox)),
        within,
    )


def filter_tiles(tiles, geometry, within: bool = False) -> list:
    """Filter tiles by intersection or containment with a geometry."""
    result = []
    geom_shape = shape(geometry)
    for tile in tiles:
        tile_shape = shape(mercantile.feature(tile)["geometry"])
        if within:
            if tile_shape.within(geom_shape):
                result.append(tile)
        elif tile_shape.intersects(geom_shape):
            result.append(tile)
    return result


def split_geojson_by_tiles(
    mother_geojson_path: str,
    children_geojson_path: str,
    output_dir: str,
    prefix: str = "OAM",
    burn_to_raster: bool = False,
    burn_value: int = 255,
) -> None:
    """Split a GeoJSON by tile geometries, optionally burning to raster masks."""
    mother_data = gpd.read_file(mother_geojson_path)

    with open(children_geojson_path, encoding="utf-8") as f:
        tiles = json.load(f)

    for tile in tiles["features"]:
        tile_geom = shape(tile["geometry"])
        tile_props = tile.get("properties", {})
        title = tile_props.get("title", "")

        x, y, z = _parse_tile_coords(title, tile)
        tile_filename = f"{prefix}-{x}-{y}-{z}"

        clipped_data = mother_data[mother_data.intersects(tile_geom)].copy()
        clipped_data = gpd.clip(clipped_data, tile_geom)
        os.makedirs(os.path.join(output_dir, "geojson"), exist_ok=True)

        clipped_filename = os.path.join(output_dir, "geojson", f"{tile_filename}.geojson")
        clipped_data.to_file(clipped_filename, driver="GeoJSON", encoding="utf-8")

        if burn_to_raster:
            _burn_tile_to_raster(clipped_data, tile_geom, output_dir, tile_filename, burn_value)


def _parse_tile_coords(title: str, tile_feature: dict) -> tuple[int, int, int]:
    """Extract x, y, z tile coordinates from mercantile feature properties."""
    tile_id = tile_feature.get("properties", {}).get("id", tile_feature.get("id", ""))
    tile_id_str = str(tile_id)

    if "Tile(" in tile_id_str or "x=" in tile_id_str:
        import re

        match = re.search(r"x=(\d+),\s*y=(\d+),\s*z=(\d+)", tile_id_str)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))

    if title and "-" in title:
        parts = title.split("-")
        if len(parts) >= 3:
            return int(parts[-3]), int(parts[-2]), int(parts[-1])

    raise ValueError(f"Cannot parse tile coordinates from: {tile_id_str}")


def _burn_tile_to_raster(
    clipped_data: gpd.GeoDataFrame,
    tile_geom,
    output_dir: str,
    tile_filename: str,
    burn_value: int,
) -> None:
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    tif_path = os.path.join(output_dir, "mask", f"{tile_filename}.tif")

    minx, miny, maxx, maxy = tile_geom.bounds
    width, height = 256, 256
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    if not clipped_data.empty:
        shapes = [(geom, burn_value) for geom in clipped_data.geometry]
        mask = rio_features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            nodata=0,
            default_value=burn_value,
            dtype=np.uint8,
        )
    else:
        mask = np.zeros((height, width), dtype=np.uint8)

    import rasterio

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        transform=transform,
        crs=clipped_data.crs,
    ) as dst:
        dst.write(mask, 1)
