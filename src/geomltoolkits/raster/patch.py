import json
import os

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, mapping

from .._logging import get_logger, track

log = get_logger(__name__)


def create_patches(
    input_tiff: str,
    output_dir: str,
    patch_size: int = 256,
    prefix: str = "patch",
) -> str:
    """Create georeferenced patches (chips) from a COG/GeoTIFF.

    Reads the raster in non-overlapping windows of `patch_size` x `patch_size`
    pixels and writes each as a separate GeoTIFF. Also emits a
    `tile_geometries.geojson` with each patch footprint.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(input_tiff) as src:
        crs = src.crs
        src_transform = src.transform
        width, height = src.width, src.height
        profile = src.profile.copy()

        cols = range(0, width, patch_size)
        rows = range(0, height, patch_size)

        features = []
        patch_count = 0

        for row_off in track(rows, description="Creating patches...", total=len(range(0, height, patch_size))):
            for col_off in cols:
                actual_width = min(patch_size, width - col_off)
                actual_height = min(patch_size, height - row_off)

                if actual_width < patch_size or actual_height < patch_size:
                    continue

                window = Window(col_off, row_off, actual_width, actual_height)  # type: ignore[too-many-positional-arguments]
                data = src.read(window=window)

                if np.all(data == 0):
                    continue

                window_transform = rasterio.windows.transform(window, src_transform)

                patch_profile = profile.copy()
                patch_profile.update(
                    {
                        "width": actual_width,
                        "height": actual_height,
                        "transform": window_transform,
                    }
                )

                patch_filename = f"{prefix}-{col_off}-{row_off}-{patch_size}.tif"
                patch_path = os.path.join(output_dir, patch_filename)

                with rasterio.open(patch_path, "w", **patch_profile) as dst:
                    dst.write(data)

                bounds = rasterio.transform.array_bounds(actual_height, actual_width, window_transform)
                feature = {
                    "type": "Feature",
                    "properties": {
                        "filename": patch_filename,
                        "col_off": col_off,
                        "row_off": row_off,
                        "patch_size": patch_size,
                    },
                    "geometry": mapping(box(*bounds)),
                }
                features.append(feature)
                patch_count += 1

    geojson_path = os.path.join(output_dir, "tile_geometries.geojson")
    feature_collection = {"type": "FeatureCollection", "features": features}
    if crs:
        feature_collection["crs"] = {
            "type": "name",
            "properties": {"name": str(crs)},
        }

    with open(geojson_path, "w") as f:
        json.dump(feature_collection, f)

    log.info(f"Created {patch_count} patches in {output_dir}")
    return output_dir
