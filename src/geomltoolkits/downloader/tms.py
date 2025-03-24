import argparse
import asyncio
import json
import os
from typing import List, Optional, Union

import aiohttp
import geopandas as gpd
import mercantile
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds
from tqdm import tqdm

from ..utils import get_tiles


async def download_tile(
    session: aiohttp.ClientSession,
    tile_id: mercantile.Tile,
    tms: str,
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
        tms: Tile map service URL template
        out_path: Output directory for the tile
        georeference: Whether to add georeference metadata
        prefix: Prefix for output filename
        crs: Coordinate reference system (4326 or 3857)
    """
    tile_url = tms.format(z=tile_id.z, x=tile_id.x, y=tile_id.y)
    async with session.get(tile_url) as response:
        if response.status != 200:
            print(f"Error fetching tile {tile_id}: {response.status}")
            return

        tile_data = await response.content.read()
        tile_filename = f"{prefix}-{tile_id.x}-{tile_id.y}-{tile_id.z}.tif"
        tile_path = os.path.join(out_path, tile_filename)

        with open(tile_path, "wb") as f:
            f.write(tile_data)

        if georeference:
            bounds = mercantile.bounds(tile_id)
            
            # Handle different coordinate systems
            if crs == "3857":
                # Convert bounds from WGS84 to Web Mercator
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                xmin, ymin = transformer.transform(bounds.west, bounds.south)
                xmax, ymax = transformer.transform(bounds.east, bounds.north)
                mercator_bounds = (xmin, ymin, xmax, ymax)
                
                with rasterio.Env(CPL_DEBUG=False):
                    with rasterio.open(tile_path, "r+") as dataset:
                        transform = from_bounds(*mercator_bounds, dataset.width, dataset.height)
                        dataset.transform = transform
                        dataset.update_tags(ns="rio_georeference", georeferencing_applied="True")
                        dataset.crs = rasterio.crs.CRS.from_epsg(3857)
            else:  # Default to 4326
                with rasterio.Env(CPL_DEBUG=False):
                    with rasterio.open(tile_path, "r+") as dataset:
                        transform = from_bounds(*bounds, dataset.width, dataset.height)
                        dataset.transform = transform
                        dataset.update_tags(ns="rio_georeference", georeferencing_applied="True")
                        dataset.crs = rasterio.crs.CRS.from_epsg(4326)


async def download_tiles(
    tms: str,
    zoom: int,
    out: str = os.getcwd(),
    geojson: Optional[Union[str, dict]] = None,
    bbox: Optional[List[float]] = None,
    within: bool = False,
    georeference: bool = False,
    dump: bool = False,
    prefix: str = "OAM",
    crs: str = "4326",
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
    # Ensure output directories exist
    chips_dir = os.path.join(out, "chips")
    os.makedirs(chips_dir, exist_ok=True)

    # Get tiles based on input geometry
    tiles = get_tiles(geojson=geojson, bbox=bbox, zoom=zoom, within=within)
    print(f"Total tiles fetched: {len(tiles)}")

    if dump:
        feature_collection = {
            "type": "FeatureCollection",
            "features": [mercantile.feature(tile) for tile in tiles],
        }
        
        # For 3857, we need to reproject the geometries, not just set metadata
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
        tasks = [
            asyncio.create_task(
                download_tile(
                    session,
                    tile_id,
                    tms,
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
    """
    Command-line interface for tile downloading.
    """
    parser = argparse.ArgumentParser(description="Download tiles from a GeoJSON.")

    # Mutually exclusive input group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--aoi", type=str, help="Path to the GeoJSON file or GeoJSON string or bbox."
    )
    group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        help="Bounding box coordinates.",
    )

    # Required and optional arguments
    parser.add_argument(
        "--tms", required=True, help="TMS URL template for downloading tiles."
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

    # Parse arguments and run
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
        )

    asyncio.run(run())


if __name__ == "__main__":
    main()
