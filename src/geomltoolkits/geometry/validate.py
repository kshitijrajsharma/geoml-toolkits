import json
import os

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from .._logging import get_logger
from .tiles import load_geojson

log = get_logger(__name__)


def validate_polygon_geometries(
    input_geojson: str | dict,
    output_path: str | None = None,
) -> dict:
    """Validate and clean polygon geometries, removing invalid or non-polygon features."""
    if isinstance(input_geojson, str) and os.path.isfile(input_geojson):
        gdf = gpd.read_file(input_geojson)
    else:
        geojson_data = load_geojson(input_geojson)
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

    if len(gdf) < 1:
        raise ValueError("Empty file - no geometries provided")

    input_count = len(gdf)
    valid_rows = []
    removed_count = 0

    for _, row in gdf.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            removed_count += 1
            continue

        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            removed_count += 1
            continue

        if not geom.is_valid:
            try:
                fixed_geom = geom.buffer(0)
                if fixed_geom.is_valid and not fixed_geom.is_empty:
                    row = row.copy()
                    row.geometry = fixed_geom
                else:
                    removed_count += 1
                    continue
            except Exception:
                removed_count += 1
                continue

        valid_rows.append(row)

    valid_count = len(valid_rows)
    log.info(f"Input features: {input_count}, valid: {valid_count}, removed: {removed_count}")

    if valid_count < 1:
        raise ValueError("No valid geometries remaining after validation")

    valid_gdf = gpd.GeoDataFrame(valid_rows).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        valid_gdf.to_file(output_path, driver="GeoJSON")

    geojson_str = valid_gdf.to_json()
    if geojson_str is None:
        raise ValueError("Failed to serialize GeoDataFrame to GeoJSON")
    return json.loads(geojson_str)


def fix_geom(geom):
    """Fix geometry issues: convert LineStrings to Polygons, remove holes, fix invalid geometries."""
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type in ("LineString", "MultiLineString"):
        geom = _linestring_to_holefree_polygon(geom)

    if geom.geom_type == "Polygon":
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = Polygon(geom.exterior)

    elif geom.geom_type == "MultiPolygon":
        new_polys = []
        for p in geom.geoms:
            if not p.is_valid:
                p = p.buffer(0)
            new_polys.append(Polygon(p.exterior))
        geom = unary_union(new_polys)

    return geom


def filter_gdf_by_area(gdf: gpd.GeoDataFrame, min_area: float) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame to remove features smaller than min_area (square meters)."""
    orig_crs = gdf.crs
    gdf_proj = gdf.to_crs("EPSG:3857") if orig_crs is None or orig_crs.is_geographic else gdf.copy()

    gdf_proj["_area_m2"] = gdf_proj.area
    gdf_proj = gdf_proj[gdf_proj["_area_m2"] >= min_area].copy()

    if gdf_proj.crs != orig_crs:
        gdf_proj = gdf_proj.to_crs(orig_crs)

    return gdf_proj.drop(columns=["_area_m2"])


def _linestring_to_holefree_polygon(geom: LineString | MultiLineString) -> Polygon | MultiPolygon:
    if geom.is_empty:
        return geom

    if geom.geom_type == "LineString":
        coords = list(geom.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return Polygon(poly.exterior)

    polys = list(polygonize(geom))
    if not polys:
        return geom

    unioned = unary_union(polys)
    if not unioned.is_valid:
        unioned = unioned.buffer(0)

    if unioned.geom_type == "Polygon":
        return Polygon(unioned.exterior)
    if unioned.geom_type == "MultiPolygon":
        return unary_union([Polygon(p.exterior) for p in unioned.geoms])
    return unioned
