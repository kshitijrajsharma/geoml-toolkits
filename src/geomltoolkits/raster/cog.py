"""Bidirectional conversion between TMS-style tiles and Cloud-Optimized GeoTIFFs."""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from .._logging import get_logger

log = get_logger(__name__)

_RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
}


def tiles_to_cog(
    tiles: str | Path | list[str | Path],
    out_path: str | Path,
    *,
    profile: str = "deflate",
    blocksize: int = 256,
    overview_resampling: str = "average",
    indexes: list[int] | None = None,
    threads: int = 8,
) -> Path:
    """Stitch many tiles into a single Cloud-Optimized GeoTIFF.

    Inputs must share CRS and resolution (true for TMS chips at a fixed zoom).
    Pass `indexes` to pin specific bands when tiles have mixed band counts —
    e.g. TiTiler returns RGB for opaque tiles and RGBA at no-data edges; use
    `indexes=[1, 2, 3]` to read RGB only and ignore alpha.

    Phase 1 reads each tile's bounds in parallel, allocates a tiled GTiff at
    the union extent, then windowed-writes every tile into its destination
    position. Phase 2 transcodes that GTiff to COG with internal overviews
    via rio-cogeo's `cog_translate`.
    """
    paths = _collect_tile_paths(tiles)
    if not paths:
        raise FileNotFoundError(f"No tiles found at {tiles}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info("Stitching %d tiles -> %s", len(paths), out)

    with rasterio.open(paths[0]) as t0:
        ref_res = (t0.res[0], t0.res[1])
        ref_crs = t0.crs
        ref_dtype = t0.dtypes[0]
        read_indexes = indexes if indexes is not None else list(range(1, t0.count + 1))
        ref_count = len(read_indexes)

    def _bounds(p: str):
        with rasterio.open(p) as t:
            return t.bounds

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        bounds_list = list(ex.map(_bounds, paths))
    log.info("read %d tile bounds in %.1fs", len(paths), time.perf_counter() - t0)

    left = min(b.left for b in bounds_list)
    bottom = min(b.bottom for b in bounds_list)
    right = max(b.right for b in bounds_list)
    top = max(b.top for b in bounds_list)
    width = round((right - left) / ref_res[0])
    height = round((top - bottom) / ref_res[1])
    transform = from_origin(left, top, ref_res[0], ref_res[1])

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        t1 = time.perf_counter()
        log.info(
            "phase 1/2: writing %dx%d tiled GTiff (crs=%s, dtype=%s, bands=%d)",
            width,
            height,
            ref_crs,
            ref_dtype,
            ref_count,
        )
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=ref_count,
            dtype=ref_dtype,
            crs=ref_crs,
            transform=transform,
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            compress="lzw",
            photometric="rgb",
            BIGTIFF="YES",
            NUM_THREADS="ALL_CPUS",
        ) as dst:
            for p, b in zip(paths, bounds_list, strict=True):
                with rasterio.open(p) as src:
                    win = from_bounds(b.left, b.bottom, b.right, b.top, transform=transform)
                    dst.write(src.read(read_indexes), window=win)
        t2 = time.perf_counter()
        log.info("phase 1 done in %.1fs (temp %.1f MB)", t2 - t1, tmp_path.stat().st_size / 1e6)

        log.info("phase 2/2: transcoding -> COG (%s, %s overviews)", profile, overview_resampling)
        cog_kwargs = cog_profiles.get(profile)
        cog_kwargs.update({"blockxsize": blocksize, "blockysize": blocksize, "BIGTIFF": "YES"})
        cog_translate(
            str(tmp_path),
            str(out),
            cog_kwargs,
            overview_resampling=overview_resampling,
            use_cog_driver=True,
            quiet=True,
        )
        log.info("phase 2 done in %.1fs", time.perf_counter() - t2)
    finally:
        tmp_path.unlink(missing_ok=True)

    log.info("COG written: %s (%.2f MB, %.1fs total)", out, out.stat().st_size / 1e6, time.perf_counter() - t0)
    return out


def cog_to_tiles(
    cog_path: str | Path,
    zoom: int,
    out_dir: str | Path,
    *,
    bbox_4326: tuple[float, float, float, float] | None = None,
    resampling: str = "bilinear",
    tile_scheme: str = "xyz",
    threads: int = 8,
) -> Path:
    """Render a COG into 256x256 WebMercator chips at a given zoom level.

    Output layout `{out_dir}/{z}/{x}/{y}.tif` (xyz; tms also supported).
    Mirrors what a TiTiler /cog/tiles/{z}/{x}/{y} endpoint returns.
    """
    cog = Path(cog_path)
    out_dir = Path(out_dir)
    if tile_scheme not in ("xyz", "tms"):
        raise ValueError(f"tile_scheme must be 'xyz' or 'tms', got {tile_scheme!r}")
    if resampling not in _RESAMPLING:
        raise ValueError(f"resampling must be one of {sorted(_RESAMPLING)}, got {resampling!r}")
    resamp = _RESAMPLING[resampling]

    with rasterio.open(cog) as src:
        if bbox_4326 is None:
            bbox_4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        src_count = src.count
        src_dtype = src.dtypes[0]

    tiles = list(mercantile.tiles(*bbox_4326, zoom))
    log.info("Rendering %d tiles at z=%d -> %s", len(tiles), zoom, out_dir)

    def _render(t: mercantile.Tile) -> None:
        b = mercantile.xy_bounds(t)
        y = t.y if tile_scheme == "xyz" else (2**zoom - 1 - t.y)
        tile_path = out_dir / str(zoom) / str(t.x) / f"{y}.tif"
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        dst_tf = from_origin(b.left, b.top, (b.right - b.left) / 256, (b.top - b.bottom) / 256)
        dst_arr = np.zeros((src_count, 256, 256), dtype=src_dtype)
        with rasterio.open(cog) as src:
            for i in range(src_count):
                reproject(
                    source=rasterio.band(src, i + 1),
                    destination=dst_arr[i],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_tf,
                    dst_crs="EPSG:3857",
                    resampling=resamp,
                )
        with rasterio.open(
            tile_path,
            "w",
            driver="GTiff",
            width=256,
            height=256,
            count=src_count,
            dtype=src_dtype,
            crs="EPSG:3857",
            transform=dst_tf,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
        ) as dst:
            dst.write(dst_arr)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(_render, tiles))
    log.info(
        "Rendered %d tiles in %.1fs (%.1f tiles/s)",
        len(tiles),
        time.perf_counter() - t0,
        len(tiles) / max(time.perf_counter() - t0, 1e-6),
    )
    return out_dir


def _collect_tile_paths(tiles: str | Path | list[str | Path]) -> list[str]:
    if isinstance(tiles, (str, Path)):
        p = Path(tiles)
        if p.is_dir():
            return sorted([str(x) for x in p.rglob("*.tif")] + [str(x) for x in p.rglob("*.tiff")])
        return [str(p)]
    return [str(x) for x in tiles]
