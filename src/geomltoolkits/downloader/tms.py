import asyncio
import json
import os
from typing import Any

import aiohttp
import geopandas as gpd
import mercantile
import rasterio
from rasterio.transform import from_bounds

from .._logging import get_logger, track
from ..geometry.crs import create_transformer, detect_scheme_from_url
from ..geometry.tiles import get_tiles

log = get_logger(__name__)


async def fetch_tilejson(session: aiohttp.ClientSession, tilejson_url: str) -> dict[str, Any]:
    async with session.get(tilejson_url) as response:
        if response.status != 200:
            raise ValueError(f"Failed to fetch TileJSON from {tilejson_url}: {response.status}")
        return await response.json()


class TileSource:
    def __init__(
        self,
        url: str,
        scheme: str = "xyz",
        fformat: str | None = "tif",
        min_zoom: int = 2,
        max_zoom: int = 18,
    ):
        self.url = url
        self.scheme = scheme.lower()
        self.fformat = fformat
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.tilejson = None

    @classmethod
    async def from_tilejson(cls, session: aiohttp.ClientSession, tilejson_url: str):
        tilejson = await fetch_tilejson(session, tilejson_url)

        source = cls(url="")
        source.tilejson = tilejson
        source.min_zoom = tilejson.get("minzoom", 2)
        source.max_zoom = tilejson.get("maxzoom", 18)

        if tilejson.get("tiles"):
            source.url = tilejson["tiles"][0]
        else:
            raise ValueError("No tile URLs found in TileJSON")

        source.scheme = tilejson.get("scheme", "xyz").lower()

        if "format" in tilejson:
            source.fformat = tilejson["format"]
        elif "{format}" in source.url:
            source.fformat = "png"
            source.url = source.url.replace("{format}", source.fformat)

        return source

    def get_tile_url(self, tile: mercantile.Tile) -> str:
        if self.scheme == "xyz":
            try:
                return self.url.format(z=tile.z, x=tile.x, y=tile.y)
            except KeyError as exc:
                if "{-y}" in self.url:
                    return self.url.format(z=tile.z, x=tile.x).replace("{-y}", str(-tile.y))
                raise ValueError(f"Unsupported XYZ format: {self.url}") from exc
        elif self.scheme == "tms":
            y_tms = (2**tile.z) - 1 - tile.y
            return self.url.format(z=tile.z, x=tile.x, y=y_tms)
        elif self.scheme == "quadkey":
            quadkey = mercantile.quadkey(tile)
            return self.url.format(q=quadkey, s=str((tile.x + tile.y) % 4))
        elif self.scheme == "mapbox":
            subdomain_letters = ["a", "b", "c", "d"]
            subdomain = subdomain_letters[(tile.x + tile.y) % 4]
            return self.url.format(z=tile.z, x=tile.x, y=tile.y, s=subdomain)
        elif self.scheme == "custom":
            return (
                self.url.format(z=tile.z, x=tile.x, y=tile.y, q=mercantile.quadkey(tile))
                .replace("{-y}", str((2**tile.z) - 1 - tile.y))
                .replace("{2^z}", str(2**tile.z))
            )
        elif self.scheme == "wms":
            bounds = mercantile.bounds(tile)
            return self.url.format(
                bbox=f"{bounds.south},{bounds.west},{bounds.north},{bounds.east}",
                width=256,
                height=256,
                proj="EPSG:4326",
            )
        elif self.scheme == "wmts":
            return self.url.format(TileMatrix=tile.z, TileCol=tile.x, TileRow=tile.y)
        raise ValueError(f"Unsupported tile scheme: {self.scheme}")

    def is_valid_zoom(self, zoom: int) -> bool:
        return self.min_zoom <= zoom <= self.max_zoom


async def download_tile(
    session: aiohttp.ClientSession,
    tile_id: mercantile.Tile,
    tile_source: TileSource,
    out_path: str,
    georeference: bool = False,
    prefix: str = "OAM",
    crs: str = "4326",
    extension: str = "tif",
) -> str | None:
    """Download a single tile asynchronously."""
    tile_url = tile_source.get_tile_url(tile_id)

    try:
        async with session.get(tile_url) as response:
            if response.status != 200:
                log.warning(f"Failed to fetch tile {tile_id}: HTTP {response.status}")
                return None

            tile_data = await response.content.read()
            tile_filename = f"{prefix}-{tile_id.x}-{tile_id.y}-{tile_id.z}.{extension}"
            tile_path = os.path.join(out_path, tile_filename)

            with open(tile_path, "wb") as f:
                f.write(tile_data)

            if georeference and extension.lower() in ("tif", "tiff"):
                _georeference_downloaded_tile(tile_id, tile_path, crs)

            return tile_path
    except Exception as e:
        log.warning(f"Error downloading tile {tile_id}: {e}")
        return None


def _georeference_downloaded_tile(tile_id: mercantile.Tile, tile_path: str, crs: str) -> None:
    bounds = mercantile.bounds(tile_id)

    try:
        with rasterio.open(tile_path, "r+") as dataset:
            if crs == "3857":
                transformer = create_transformer(4326, 3857)
                xmin, ymin = transformer.transform(bounds.west, bounds.south)
                xmax, ymax = transformer.transform(bounds.east, bounds.north)
                transform = from_bounds(xmin, ymin, xmax, ymax, dataset.width, dataset.height)
                dataset.crs = rasterio.crs.CRS.from_epsg(3857)
            else:
                transform = from_bounds(
                    bounds.west, bounds.south, bounds.east, bounds.north, dataset.width, dataset.height
                )
                dataset.crs = rasterio.crs.CRS.from_epsg(4326)

            dataset.transform = transform
            dataset.update_tags(ns="rio_georeference", georeferencing_applied="True")
    except rasterio.errors.RasterioIOError:
        log.warning(f"Could not georeference {tile_path}: not a valid raster file")


async def download_tiles(
    tms: str | TileSource,
    zoom: int,
    out: str = ".",
    geojson: str | dict | None = None,
    bbox: list[float] | None = None,
    within: bool = False,
    georeference: bool = False,
    dump_tile_geometries_as_geojson: bool = False,
    prefix: str = "OAM",
    crs: str = "4326",
    tile_scheme: str | None = None,
    is_tilejson: bool = False,
    extension: str = "tif",
    max_failures: int | None = None,
) -> str:
    """Download tiles from a GeoJSON or bounding box asynchronously."""
    chips_dir = os.path.join(out, "chips")
    os.makedirs(chips_dir, exist_ok=True)
    tiles = get_tiles(zoom=zoom, geojson=geojson, bbox=bbox, within=within)
    log.info(f"Total tiles to download: {len(tiles)}")

    if dump_tile_geometries_as_geojson:
        _dump_tile_geometries(tiles, out, crs)

    if not tile_scheme:
        tile_scheme = tms.scheme if isinstance(tms, TileSource) else detect_scheme_from_url(tms)
        log.info(f"Detected tile scheme: {tile_scheme}")

    async with aiohttp.ClientSession() as session:
        tile_source = await _resolve_tile_source(session, tms, tile_scheme, is_tilejson)

        if not tile_source.is_valid_zoom(zoom):
            log.warning(f"Zoom level {zoom} is outside supported range ({tile_source.min_zoom}-{tile_source.max_zoom})")

        tasks = [
            asyncio.create_task(
                download_tile(session, tile_id, tile_source, chips_dir, georeference, prefix, crs, extension)
            )
            for tile_id in tiles
        ]

        results = []
        for coro in track(asyncio.as_completed(tasks), description="Downloading tiles...", total=len(tasks)):
            results.append(await coro)

    failures = sum(1 for r in results if r is None)
    if failures:
        log.warning(f"{failures}/{len(tiles)} tiles failed to download")
    if max_failures is not None and failures > max_failures:
        raise RuntimeError(f"Too many tile download failures: {failures} > {max_failures}")

    return chips_dir


async def _resolve_tile_source(
    session: aiohttp.ClientSession,
    tms: str | TileSource,
    tile_scheme: str,
    is_tilejson: bool,
) -> TileSource:
    if isinstance(tms, TileSource):
        return tms
    if is_tilejson:
        return await TileSource.from_tilejson(session, tms)
    return TileSource(tms, scheme=tile_scheme)


def _dump_tile_geometries(tiles: list, out: str, crs: str) -> None:
    feature_collection = {
        "type": "FeatureCollection",
        "features": [mercantile.feature(tile) for tile in tiles],
    }

    if crs == "3857":
        gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])
        gdf.set_crs(epsg=4326, inplace=True)
        gdf = gdf.to_crs(epsg=3857)
        feature_collection = json.loads(gdf.to_json())
        feature_collection["crs"] = {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::3857"},
        }
    else:
        feature_collection["crs"] = {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
        }

    with open(os.path.join(out, "tiles.geojson"), "w") as f:
        json.dump(feature_collection, f)
