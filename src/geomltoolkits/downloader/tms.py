import argparse
import asyncio
import json
import os
import re
import mimetypes
from typing import Dict, List, Optional, Union, Any

import aiohttp
import geopandas as gpd
import mercantile
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds
from tqdm import tqdm

from ..utils import get_tiles


async def fetch_tilejson(session: aiohttp.ClientSession, tilejson_url: str) -> Dict[str, Any]:
    async with session.get(tilejson_url) as response:
        if response.status != 200:
            raise ValueError(f"Failed to fetch TileJSON from {tilejson_url}: {response.status}")
        return await response.json()


class TileSource:
    def __init__(
        self, 
        url: str, 
        scheme: str = "xyz",
        format: Optional[str] = 'tif',
        min_zoom: int = 2,
        max_zoom: int = 18
    ):
        self.url = url
        self.scheme = scheme.lower()
        self.format = format
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.tilejson = None

    @classmethod
    async def from_tilejson(cls, session: aiohttp.ClientSession, tilejson_url: str):
        tilejson = await fetch_tilejson(session, tilejson_url)
        # my reference is this tilejson spec : https://github.com/mapbox/tilejson-spec/tree/master/3.0.0
        
        source = cls(url="")
        source.tilejson = tilejson
        source.min_zoom = tilejson.get("minzoom", 2)
        source.max_zoom = tilejson.get("maxzoom", 18)
        
        if "tiles" in tilejson and tilejson["tiles"]:
            source.url = tilejson["tiles"][0] # currently only single tile source is supported , if any situation comesup in future we will handle it accordingly
        else:
            raise ValueError("No tile URLs found in TileJSON")
        
        source.scheme = tilejson.get("scheme", "xyz").lower()
        
        if "format" in tilejson:
            source.format = tilejson["format"]
        elif "{format}" in source.url:
            source.format = "png"
            source.url = source.url.replace("{format}", source.format)
            
        return source
    
    def get_tile_url(self, tile: mercantile.Tile) -> str:
        if self.scheme == "xyz": # source : https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
            try :
                return self.url.format(z=tile.z, x=tile.x, y=tile.y)
            except KeyError: # possible inverted xyz tiles
                if "{-y}" in self.url:
                    return self.url.format(z=tile.z, x=tile.x).replace("{-y}",-tile.y)
                else:
                    raise ValueError(f"Unsupported XYZ format: {self.url}")
        elif self.scheme == "tms": # source : https://wiki.osgeo.org/wiki/Tile_Map_Service_Specification
            y_tms = (2**tile.z) - 1 - tile.y # Converting XYZ y to TMS y
            return self.url.format(z=tile.z, x=tile.x, y=y_tms)
        
        elif self.scheme == "quadkey":
            quadkey = mercantile.quadkey(tile)
            return self.url.format(q=quadkey)
        
        elif self.scheme == "custom":
            return self.url.format(
                z=tile.z, 
                x=tile.x, 
                y=tile.y,
                q=mercantile.quadkey(tile)
            ).replace("{-y}", str((2**tile.z) - 1 - tile.y)).replace("{2^z}", str(2**tile.z))

        else:
            raise ValueError(f"Unsupported tile scheme: {self.scheme}")
    
    def is_valid_zoom(self, zoom: int) -> bool:
        return self.min_zoom <= zoom <= self.max_zoom


async def download_tile(
    session: aiohttp.ClientSession,
    tile_id: mercantile.Tile,
    tile_source: Union[TileSource, str],
    out_path: str,
    georeference: bool = False,
    prefix: str = "OAM",
    crs: str = "4326",
) -> None:
    """
    Download a single tile asynchronously.

    Args:
        session: Active aiohttp client session
        tile_id: Mercantile tile to download
        tile_source: Tile map service URL template
        out_path: Output directory for the tile
        georeference: Whether to add georeference metadata
        prefix: Prefix for output filename
        crs: Coordinate reference system (4326 or 3857)
    """
    if isinstance(tile_source, str):
        try:
            tile_url = tile_source.format(z=tile_id.z, x=tile_id.x, y=tile_id.y)
        except KeyError:
            try:
                if "{-y}" in tile_source:
                    return tile_source.format(z=tile_id.z, x=tile_id.x).replace("{-y}",-tile_id.y)
                else:
                    raise ValueError(f"Unsupported XYZ format: {tile_source}")
                # y_tms = (2**tile_id.z) - 1 - tile_id.y
            except KeyError:
                try:
                    quadkey = mercantile.quadkey(tile_id)
                    tile_url = tile_source.format(q=quadkey)
                except KeyError:
                    raise ValueError(f"Unsupported URL template format: {tile_source}")
    else:
        tile_url = tile_source.get_tile_url(tile_id)
    
    async with session.get(tile_url) as response:
        if response.status != 200:
            print(f"Error fetching tile {tile_id}: {response.status}")
            return

   
        extension = 'tif'
        
        tile_data = await response.content.read()
        tile_filename = f"{prefix}-{tile_id.x}-{tile_id.y}-{tile_id.z}.{extension}"
        tile_path = os.path.join(out_path, tile_filename)

        with open(tile_path, "wb") as f:
            f.write(tile_data)

        if georeference and extension.lower() in ['tif', 'tiff']:
            bounds = mercantile.bounds(tile_id)
            
            if crs == "3857":
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                xmin, ymin = transformer.transform(bounds.west, bounds.south)
                xmax, ymax = transformer.transform(bounds.east, bounds.north)
                mercator_bounds = (xmin, ymin, xmax, ymax)
                
                with rasterio.Env(CPL_DEBUG=False):
                    try:
                        with rasterio.open(tile_path, "r+") as dataset:
                            transform = from_bounds(*mercator_bounds, dataset.width, dataset.height)
                            dataset.transform = transform
                            dataset.update_tags(ns="rio_georeference", georeferencing_applied="True")
                            dataset.crs = rasterio.crs.CRS.from_epsg(3857)
                    except rasterio.errors.RasterioIOError:
                        print(f"Warning: Could not georeference {tile_path}. Not a valid raster file.")
            else:
                with rasterio.Env(CPL_DEBUG=False):
                    try:
                        with rasterio.open(tile_path, "r+") as dataset:
                            transform = from_bounds(*bounds, dataset.width, dataset.height)
                            dataset.transform = transform
                            dataset.update_tags(ns="rio_georeference", georeferencing_applied="True")
                            dataset.crs = rasterio.crs.CRS.from_epsg(4326)
                    except rasterio.errors.RasterioIOError:
                        print(f"Warning: Could not georeference {tile_path}. Not a valid raster file.")


async def download_tiles(
    tms: Union[str, TileSource],
    zoom: int,
    out: str = os.getcwd(),
    geojson: Optional[Union[str, dict]] = None,
    bbox: Optional[List[float]] = None,
    within: bool = False,
    georeference: bool = False,
    dump: bool = False,
    prefix: str = "OAM",
    crs: str = "4326",
    tile_scheme: str = "xyz",
    is_tilejson: bool = False,
) -> None:
    """
    Download tiles from a GeoJSON or bounding box asynchronously.

    Args:
        tms: Tile map service URL template
        zoom: Zoom level for tiles
        out: Output directory for downloaded tiles
        geojson: GeoJSON file path, string, or dictionary
        bbox: Bounding box coordinates
        within: Download only tiles completely within geometry
        georeference: Add georeference metadata to tiles
        dump: Dump tile geometries to a GeoJSON file
        prefix: Prefix for output filenames
        crs: Coordinate reference system (4326 or 3857)
    """
    chips_dir = os.path.join(out, "chips")
    os.makedirs(chips_dir, exist_ok=True)

    tiles = get_tiles(zoom=zoom, geojson=geojson, bbox=bbox, within=within)
    print(f"Total tiles fetched: {len(tiles)}")

    if dump:
        feature_collection = {
            "type": "FeatureCollection",
            "features": [mercantile.feature(tile) for tile in tiles],
        }
        
        if crs == "3857":
            gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])
            
            gdf.set_crs(epsg=4326, inplace=True)
            gdf = gdf.to_crs(epsg=3857)
            reprojected_fc = json.loads(gdf.to_json())
            feature_collection = reprojected_fc
            
            feature_collection["crs"] = {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}
            }
        else:
            feature_collection["crs"] = {
                "type": "name", 
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}
            }
            
        with open(os.path.join(out, "tiles.geojson"), "w") as f:
            json.dump(feature_collection, f)
    
    async with aiohttp.ClientSession() as session:
        if isinstance(tms, str):
            if is_tilejson:
                tile_source = await TileSource.from_tilejson(session, tms)
            else:
                tile_source = TileSource(tms, scheme=tile_scheme)
        else:
            tile_source = tms
            
        if not tile_source.is_valid_zoom(zoom):
            print(f"Warning: Requested zoom level {zoom} is outside the source's supported range ({tile_source.min_zoom}-{tile_source.max_zoom})")

        tasks = [
            asyncio.create_task(
                download_tile(
                    session,
                    tile_id,
                    tile_source,
                    chips_dir,
                    georeference,
                    prefix,
                    crs,
                )
            )
            for tile_id in tiles
        ]

        pbar = tqdm(total=len(tasks), unit="tile")
        for future in asyncio.as_completed(tasks):
            await future
            pbar.update(1)
        pbar.close()
    
    return chips_dir


def main():
    parser = argparse.ArgumentParser(description="Download tiles from a GeoJSON or bounding box.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--aoi", type=str, help="Path to the GeoJSON file or GeoJSON string."
    )
    group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        help="Bounding box coordinates.",
    )

    parser.add_argument(
        "--tms", required=True, help="TMS URL template for downloading tiles."
    )
    parser.add_argument(
        "--scheme",
        choices=["xyz", "tms", "quadkey", "custom"],
        default="xyz",
        help="Tile URL scheme (default: xyz)."
    )
    parser.add_argument(
        "--tilejson",
        action="store_true",
        help="Treat the TMS URL as a TileJSON URL."
    )
    parser.add_argument("--zoom", type=int, required=True, help="Zoom level for tiles.")
    parser.add_argument(
        "--out",
        default=os.path.join(os.getcwd()),
        help="Directory to save downloaded tiles.",
    )
    parser.add_argument(
        "--within",
        action="store_true",
        help="Download only tiles completely within the GeoJSON geometry.",
    )
    parser.add_argument(
        "--georeference",
        action="store_true",
        help="Georeference the downloaded tiles using tile bounds.",
    )
    parser.add_argument(
        "--dump", action="store_true", help="Dump tile geometries to a GeoJSON file."
    )
    parser.add_argument(
        "--prefix",
        default="OAM",
        help="Prefix for output tile filenames (default: OAM).",
    )
    parser.add_argument(
        "--crs",
        choices=["4326", "3857"],
        default="4326",
        help="Coordinate reference system for georeferenced tiles (default: 4326).",
    )

    args = parser.parse_args()

    async def run():
        await download_tiles(
            geojson=args.aoi,
            bbox=args.bbox,
            tms=args.tms,
            zoom=args.zoom,
            out=args.out,
            within=args.within,
            georeference=args.georeference,
            dump=args.dump,
            prefix=args.prefix,
            crs=args.crs,
            tile_scheme=args.scheme,
            is_tilejson=args.tilejson,
        )

    asyncio.run(run())


if __name__ == "__main__":
    main()