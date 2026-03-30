import json
import math
from typing import Any

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import unary_union

from .._logging import get_logger

log = get_logger(__name__)


def detect_and_ensure_4326_geometry(geojson: str | dict) -> dict[str, Any]:
    """Detect CRS from GeoJSON, convert to EPSG:4326 if needed, and union all geometries."""
    if isinstance(geojson, str):
        if __import__("os").path.exists(geojson):
            gdf = gpd.read_file(geojson)
        else:
            try:
                geojson_data = json.loads(geojson)
                if "features" in geojson_data:
                    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
                else:
                    gdf = gpd.GeoDataFrame(geometry=[shape(json.loads(geojson))])
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid GeoJSON string provided") from exc
    else:
        if "features" in geojson:
            gdf = gpd.GeoDataFrame.from_features(geojson["features"])
        else:
            gdf = gpd.GeoDataFrame(geometry=[shape(geojson)])

    if gdf.crs is None:
        log.warning("No CRS found in GeoJSON. Assuming EPSG:4326 (WGS84).")
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        original_crs = gdf.crs.to_epsg()
        log.info(f"Converting GeoJSON from EPSG:{original_crs} to EPSG:4326 for API compatibility...")
        gdf = gdf.to_crs(epsg=4326)

    unioned_geometry = unary_union(gdf.geometry.values)
    return gpd.GeoSeries([unioned_geometry]).set_crs(epsg=4326).__geo_interface__


def reproject_geojson(geojson_data: dict[str, Any], target_crs: str) -> dict[str, Any]:
    """Reproject GeoJSON data to the specified CRS."""
    if target_crs == "4326":
        return geojson_data

    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(epsg=int(target_crs))

    reprojected_geojson = json.loads(gdf.to_json())
    reprojected_geojson["crs"] = {
        "type": "name",
        "properties": {"name": f"urn:ogc:def:crs:EPSG::{target_crs}"},
    }
    return reprojected_geojson


def ensure_crs(geojson_path: str, target_epsg: int = 4326) -> None:
    """Validate and convert a GeoJSON file to the target CRS if needed."""
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None or gdf.crs.to_epsg() != target_epsg:
        log.info(f"Converting {geojson_path} from {gdf.crs} to EPSG:{target_epsg}")
        gdf = gdf.to_crs(epsg=target_epsg)
        gdf.to_file(geojson_path, driver="GeoJSON")


def num2deg(x_tile: int, y_tile: int, zoom: int) -> tuple[float, float]:
    """Convert Web Mercator tile coordinates to lat/lon (NW corner)."""
    n = 2.0**zoom
    lon = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y_tile / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def km_to_degrees(km_distance: float, latitude: float = 0.0) -> tuple[float, float]:
    """Convert kilometers to approximate degrees at a given latitude."""
    lat_degrees = km_distance / 111.0
    lon_degrees = km_distance / (111.0 * math.cos(math.radians(latitude)))
    return lat_degrees, lon_degrees


def degrees_to_km(degrees: float, latitude: float = 0.0) -> tuple[float, float]:
    """Convert degrees to approximate kilometers at a given latitude."""
    lat_km = degrees * 111.0
    lon_km = degrees * 111.0 * math.cos(math.radians(latitude))
    return lat_km, lon_km


def detect_scheme_from_url(url: str) -> str:
    """Detect tile URL scheme from URL template pattern."""
    if "{q}" in url or "quadkey" in url.lower():
        return "quadkey"
    elif "{-y}" in url.lower():
        return "tms"
    elif "tiles.mapbox.com" in url.lower():
        return "mapbox"
    elif all(tag in url for tag in ["{z}", "{x}", "{y}"]):
        return "xyz"
    elif "service=wms" in url.lower():
        return "wms"
    elif "service=wmts" in url.lower():
        return "wmts"
    return "custom"


def create_transformer(source_epsg: int, target_epsg: int) -> Transformer:
    """Create a pyproj Transformer between two EPSG codes."""
    return Transformer.from_crs(f"EPSG:{source_epsg}", f"EPSG:{target_epsg}", always_xy=True)
