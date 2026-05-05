import asyncio
import io
import json
import os
import zipfile
from typing import Any

import aiohttp

from .._logging import get_logger
from ..geometry.crs import detect_and_ensure_4326_geometry, reproject_geojson
from ..geometry.tiles import check_geojson_geom, load_geometry, split_geojson_by_tiles

log = get_logger(__name__)


class RawDataAPI:
    """Client for interacting with the HOT Raw Data API."""

    def __init__(self, base_api_url: str = "https://api-prod.raw-data.hotosm.org/v1"):
        self.base_api_url = base_api_url
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Referer": "geomltoolkits-python-lib",
        }

    async def request_snapshot(
        self,
        geometry: dict[str, Any],
        feature_type: str = "building",
        geometry_types: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a snapshot of OSM data for a given geometry.

        ``filters``, when provided, replaces the auto-built ``tags.all_geometry.join_or``
        block. Use it to drive the raw-data API directly with multi-tag /
        per-geometry filters, e.g.
        ``{"tags": {"polygon": {"join_or": {"building": [], "amenity": ["hospital"]}}}}``.
        When ``filters`` is None, the body falls back to a single
        ``{feature_type: []}`` join_or for back-compat.
        """
        if geometry_types is None:
            geometry_types = ["polygon"]
        if filters is None:
            filters = {"tags": {"all_geometry": {"join_or": {feature_type: []}}}}

        payload = {
            "fileName": "geomltoolkits",
            "geometry": geometry,
            "filters": filters,
            "geometryType": geometry_types,
        }

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_api_url}/snapshot/",
                data=json.dumps(payload),
                headers=self.headers,
            ) as response,
        ):
            response_data = await response.json()
            try:
                response.raise_for_status()
            except Exception as ex:
                raise RuntimeError(f"API error: {json.dumps(response_data)}") from ex
            return response_data

    async def poll_task_status(
        self,
        task_link: str,
        max_wait_seconds: int = 600,
    ) -> dict[str, Any]:
        """Poll the API for task completion with a timeout."""
        elapsed = 0
        poll_interval = 2

        async with aiohttp.ClientSession() as session:
            while elapsed < max_wait_seconds:
                async with session.get(url=f"{self.base_api_url}{task_link}", headers=self.headers) as response:
                    response.raise_for_status()
                    res = await response.json()
                    if res["status"] in ("SUCCESS", "FAILED"):
                        return res
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        raise TimeoutError(f"Task did not complete within {max_wait_seconds} seconds")

    async def download_snapshot(self, download_url: str) -> dict[str, Any]:
        """Download and extract snapshot data from the provided URL."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(url=download_url, headers=self.headers) as response,
        ):
            response.raise_for_status()
            data = await response.read()
            with zipfile.ZipFile(io.BytesIO(data), "r") as zip_ref, zip_ref.open("geomltoolkits.geojson") as file:
                return json.load(file)

    async def last_updated(self) -> str:
        """Get the last updated timestamp from the API."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{self.base_api_url}/status", headers=self.headers) as response,
        ):
            response_data = await response.json()
            try:
                response.raise_for_status()
            except Exception as ex:
                raise RuntimeError(f"API error: {json.dumps(response_data)}") from ex
            return response_data["lastUpdated"]


async def download_osm_data(
    geojson: str | dict | None = None,
    bbox: list[float] | None = None,
    api_url: str = "https://api-prod.raw-data.hotosm.org/v1",
    feature_type: str = "building",
    geometry_types: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    dump_results: bool = False,
    out: str | None = None,
    split_output_by_tiles: bool = False,
    split_prefix: str = "OAM",
    crs: str = "4326",
    burn_splits_to_raster: bool = False,
    burn_value: int = 255,
    max_wait_seconds: int = 600,
) -> dict[str, Any] | str:
    """Download OSM data for a given geometry via the HOT Raw Data API.

    ``filters``: optional override for the raw-data API ``filters`` body block
    (e.g. ``{"tags": {"polygon": {"join_or": {"building": ["yes"]}}}}``). When
    set, ``feature_type`` is ignored. When unset, the request is built from
    ``feature_type`` for back-compat with simple single-tag callers.
    """
    if geojson is not None:
        geometry = detect_and_ensure_4326_geometry(geojson)
    else:
        geometry = check_geojson_geom(load_geometry(bbox=bbox))

    api = RawDataAPI(api_url)
    log.info(f"OSM Data Last Updated: {await api.last_updated()}")
    task_response = await api.request_snapshot(
        geometry, feature_type=feature_type, geometry_types=geometry_types, filters=filters
    )
    task_link = task_response.get("track_link")

    if not task_link:
        raise RuntimeError("No task link found in API response")

    result = await api.poll_task_status(task_link, max_wait_seconds)

    if result["status"] == "SUCCESS" and result["result"].get("download_url"):
        download_url = result["result"]["download_url"]
        result_geojson = await api.download_snapshot(download_url)

        if crs != "4326":
            log.info(f"Reprojecting output from EPSG:4326 to EPSG:{crs}...")
            result_geojson = reproject_geojson(result_geojson, crs)

        if dump_results and out:
            os.makedirs(out, exist_ok=True)
            output_path = os.path.join(out, "osm-result.geojson")
            log.info(f"Saving GeoJSON data (EPSG:{crs}) to: {output_path}")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_geojson, f)

            if split_output_by_tiles and isinstance(geojson, str):
                split_dir = os.path.join(out, "split")
                log.info(f"Splitting GeoJSON by tiles to: {split_dir}")
                os.makedirs(split_dir, exist_ok=True)
                split_geojson_by_tiles(
                    output_path,
                    geojson,
                    split_dir,
                    prefix=split_prefix,
                    burn_to_raster=burn_splits_to_raster,
                    burn_value=burn_value,
                )
            return out

        return result_geojson

    raise RuntimeError(f"Task failed with status: {result['status']}")
