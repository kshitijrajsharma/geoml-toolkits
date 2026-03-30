import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features as rio_features
from shapely.geometry import Polygon

from .._logging import get_logger
from ..geometry.orthogonalize import orthogonalize_gdf
from ..geometry.validate import filter_gdf_by_area, fix_geom

log = get_logger(__name__)


def vectorize_raster(input_tiff: str, threshold: float = 0) -> gpd.GeoDataFrame:
    """Vectorize a GeoTIFF binary mask into polygons using rasterio.features."""
    with rasterio.open(input_tiff) as src:
        raster = src.read(1)
        transform = src.transform
        crs = src.crs
        mask = raster > threshold
        shapes = list(rio_features.shapes(mask.astype(np.uint8), mask=mask, transform=transform))

    polygons = [Polygon(shape["coordinates"][0]) for shape, value in shapes if value == 1]

    log.info(f"Extracted {len(polygons)} polygons from {input_tiff}")
    return gpd.GeoDataFrame(geometry=polygons, crs=crs)


def vectorize_mask(
    input_tiff: str,
    output_geojson: str,
    simplify_tolerance: float = 0.2,
    min_area: float = 1.0,
    orthogonalize: bool = True,
    ortho_skew_tolerance_deg: int = 0,
    ortho_max_angle_change_deg: int = 45,
) -> gpd.GeoDataFrame:
    """Convert a GeoTIFF mask to a cleaned, simplified GeoJSON."""
    gdf = vectorize_raster(input_tiff)

    gdf["geometry"] = gdf["geometry"].apply(fix_geom)

    if simplify_tolerance > 0:
        gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance, preserve_topology=True)

    if orthogonalize:
        log.info("Orthogonalizing geometries...")
        gdf = orthogonalize_gdf(
            gdf,
            maxAngleChange=ortho_max_angle_change_deg,
            skewTolerance=ortho_skew_tolerance_deg,
        )

    if min_area > 0:
        original_count = len(gdf)
        gdf = filter_gdf_by_area(gdf, min_area)
        log.info(f"Filtered out {original_count - len(gdf)} features below {min_area} sq meters")

    if gdf.crs and gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    gdf.to_file(output_geojson, driver="GeoJSON")
    log.info(f"Saved {len(gdf)} features to {output_geojson}")
    return gdf
