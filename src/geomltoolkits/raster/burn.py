import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features as rio_features

from .._logging import get_logger, track

log = get_logger(__name__)


def burn_labels(
    labels_path: str,
    chips_dir: str,
    output_dir: str,
    burn_value: int = 255,
    class_property: str | None = None,
) -> str:
    """Rasterize GeoJSON labels onto matching chips as binary or multi-class masks.

    For each chip in `chips_dir`, clips the labels to the chip extent and
    rasterizes them. If `class_property` is provided, uses that feature
    property as the burn value for multi-class masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels_gdf = gpd.read_file(labels_path)

    chip_files = [f for f in os.listdir(chips_dir) if f.lower().endswith((".tif", ".tiff"))]

    burned_count = 0
    for chip_file in track(chip_files, description="Burning labels..."):
        chip_path = os.path.join(chips_dir, chip_file)

        with rasterio.open(chip_path) as src:
            chip_transform = src.transform
            chip_crs = src.crs
            chip_width = src.width
            chip_height = src.height

        chip_bounds = rasterio.transform.array_bounds(chip_height, chip_width, chip_transform)

        if labels_gdf.crs and labels_gdf.crs != chip_crs:
            labels_reprojected = labels_gdf.to_crs(chip_crs)
        else:
            labels_reprojected = labels_gdf

        from shapely.geometry import box

        chip_geom = box(*chip_bounds)
        clipped = labels_reprojected[labels_reprojected.intersects(chip_geom)].copy()
        clipped = gpd.clip(clipped, chip_geom)

        if clipped.empty:
            mask = np.zeros((chip_height, chip_width), dtype=np.uint8)
        else:
            if class_property and class_property in clipped.columns:
                shapes = [(geom, int(val)) for geom, val in zip(clipped.geometry, clipped[class_property], strict=True)]
            else:
                shapes = [(geom, burn_value) for geom in clipped.geometry]

            mask = rio_features.rasterize(
                shapes=shapes,
                out_shape=(chip_height, chip_width),
                transform=chip_transform,
                fill=0,
                dtype=np.uint8,
            )

        mask_filename = os.path.splitext(chip_file)[0] + ".tif"
        mask_path = os.path.join(output_dir, mask_filename)

        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=chip_height,
            width=chip_width,
            count=1,
            dtype=np.uint8,
            transform=chip_transform,
            crs=chip_crs,
        ) as dst:
            dst.write(mask, 1)

        burned_count += 1

    log.info(f"Burned labels onto {burned_count} chips in {output_dir}")
    return output_dir
