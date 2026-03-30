import os

import rasterio
from rasterio.merge import merge

from .._logging import get_logger

log = get_logger(__name__)


def merge_rasters(input_files: str | list[str], output_path: str) -> str:
    """Merge multiple raster files or a directory of TIFFs into a single output."""
    if isinstance(input_files, str):
        if os.path.isdir(input_files):
            files = []
            for root, _, filenames in os.walk(input_files):
                for f in filenames:
                    if f.lower().endswith(".tif"):
                        files.append(os.path.join(root, f))
            input_files = files
        else:
            raise ValueError("input_files must be a list or directory path")
    elif not isinstance(input_files, list):
        raise ValueError("input_files must be a list or directory path")

    src_files = [rasterio.open(fp) for fp in input_files]
    try:
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
    finally:
        for src in src_files:
            src.close()

    log.info(f"Merged {len(input_files)} rasters into {output_path}")
    return output_path
