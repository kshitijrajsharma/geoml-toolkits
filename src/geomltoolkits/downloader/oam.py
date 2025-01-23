import argparse
import asyncio
import json
import os
from typing import List, Optional, Union

import aiohttp
import mercantile
import rasterio
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
) -> None:
    """
    Download a single tile asynchronously.

    Args:
        session: Active aiohttp client session
        tile_id: Mercantile tile to download
        tms: Tile map service URL template
        out_path: Output directory for the tile
        georeference: Whether to add georeference metadata
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
            with rasterio.Env(CPL_DEBUG=False):
                with rasterio.open(
                    tile_path,
                    "r+",
                ) as dataset:
                    transform = from_bounds(*bounds, dataset.width, dataset.height)
                    dataset.transform = transform
                    dataset.update_tags(
                        ns="rio_georeference", georeferencing_applied="True"
                    )
                    dataset.crs = rasterio.crs.CRS({"init": "epsg:4326"})


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
    """
    # Ensure output directories exist
    chips_dir = os.path.join(out, "chips")
    os.makedirs(chips_dir, exist_ok=True)

    # Get tiles based on input geometry
    tiles = get_tiles(geojson=geojson, bbox=bbox, zoom=zoom, within=within)
    print(f"Total tiles fetched: {len(tiles)}")

    # Optional tile geometry dumping
    if dump:
        feature_collection = {
            "type": "FeatureCollection",
            "features": [mercantile.feature(tile) for tile in tiles],
        }
        with open(os.path.join(out, "tiles.geojson"), "w") as f:
            json.dump(feature_collection, f)
    # Download tiles
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
                )
            )
            for tile_id in tiles
        ]

        # Track download progress
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
        )

    asyncio.run(run())


if __name__ == "__main__":
    main()
