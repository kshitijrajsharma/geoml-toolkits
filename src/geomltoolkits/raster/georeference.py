import os
import re
from glob import glob

import mercantile
import rasterio
from rasterio.transform import from_bounds

from .._logging import get_logger
from ..geometry.crs import create_transformer

log = get_logger(__name__)


def georeference_tile(
    input_tiff: str,
    x: int,
    y: int,
    z: int,
    output_tiff: str,
    crs: str = "4326",
    overlap_pixels: int = 0,
    tile_size: int = 256,
    clip_bands_to: int | None = None,
) -> str:
    """Georeference a TIFF image based on tile coordinates (x, y, z) with optional overlap."""
    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    os.makedirs(os.path.dirname(os.path.abspath(output_tiff)), exist_ok=True)

    with rasterio.open(input_tiff) as src:
        kwargs = src.meta.copy()
        transform, target_crs = _compute_transform(bounds, crs, overlap_pixels, tile_size)
        band_count = min(src.count, clip_bands_to) if clip_bands_to is not None else src.count

        kwargs.update(
            {
                "driver": "GTiff",
                "crs": target_crs,
                "transform": transform,
                "height": src.height,
                "width": src.width,
                "count": band_count,
            }
        )

        with rasterio.open(output_tiff, "w", **kwargs) as dst:
            dst.write(src.read(indexes=list(range(1, band_count + 1))))
            dst.update_tags(ns="rio_georeference", georeferencing_applied="True")
            if overlap_pixels > 0:
                dst.update_tags(ns="rio_georeference", overlap_applied=str(overlap_pixels))

    return output_tiff


def _compute_transform(bounds, crs: str, overlap_pixels: int, tile_size: int):
    if crs == "3857":
        transformer = create_transformer(4326, 3857)
        xmin, ymin = transformer.transform(bounds.west, bounds.south)
        xmax, ymax = transformer.transform(bounds.east, bounds.north)
        target_crs = rasterio.CRS.from_epsg(3857)

        if overlap_pixels > 0:
            x_res = (xmax - xmin) / tile_size
            y_res = (ymax - ymin) / tile_size
            xmin -= overlap_pixels * x_res
            ymin -= overlap_pixels * y_res
            xmax += overlap_pixels * x_res
            ymax += overlap_pixels * y_res

        return from_bounds(xmin, ymin, xmax, ymax, tile_size, tile_size), target_crs

    target_crs = rasterio.CRS.from_epsg(4326)
    west, south, east, north = bounds.west, bounds.south, bounds.east, bounds.north

    if overlap_pixels > 0:
        x_res = (east - west) / tile_size
        y_res = (north - south) / tile_size
        west -= overlap_pixels * x_res
        south -= overlap_pixels * y_res
        east += overlap_pixels * x_res
        north += overlap_pixels * y_res

    return from_bounds(west, south, east, north, tile_size, tile_size), target_crs


def georeference_prediction_tiles(
    prediction_path: str,
    georeference_path: str,
    overlap_pixels: int = 0,
    crs: str = "4326",
    tile_size: int = 256,
    clip_bands_to: int | None = None,
) -> str:
    """Georeference all prediction tiles based on embedded x,y,z coordinates in filenames."""
    os.makedirs(georeference_path, exist_ok=True)

    image_files = glob(os.path.join(prediction_path, "*.png"))
    image_files.extend(glob(os.path.join(prediction_path, "*.jpeg")))

    georeferenced_count = 0

    for image_file in image_files:
        filename = os.path.basename(image_file)
        filename_without_ext = re.sub(r"\.(png|jpeg)$", "", filename)

        try:
            parts = re.split("-", filename_without_ext)
            if len(parts) < 3:
                log.warning(f"Could not extract tile coordinates from {filename}")
                continue

            x_tile, y_tile, zoom = map(int, parts[-3:])
            output_tiff = os.path.join(georeference_path, f"{filename_without_ext}.tif")

            georeference_tile(
                input_tiff=image_file,
                x=x_tile,
                y=y_tile,
                z=zoom,
                output_tiff=output_tiff,
                crs=crs,
                overlap_pixels=overlap_pixels,
                tile_size=tile_size,
                clip_bands_to=clip_bands_to,
            )
            georeferenced_count += 1
        except Exception as e:
            log.error(f"Error georeferencing {filename}: {e}")

    log.info(f"Georeferenced {georeferenced_count} tiles to {georeference_path}")
    return georeference_path
