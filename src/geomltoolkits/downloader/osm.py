import argparse
import asyncio
import io
import json
import os
import zipfile
from typing import Any, Dict, List, Optional, Union

import aiohttp

from ..utils import get_geometry, split_geojson_by_tiles


class RawDataAPI:
    """
    A client for interacting with the Humanitarian OpenStreetMap Team (HOT) Raw Data API.
    """

    def __init__(self, base_api_url: str = "https://api-prod.raw-data.hotosm.org/v1"):
        """
        Initialize the RawDataAPI with a base API URL.

        Args:
            base_api_url (str): Base URL for the Raw Data API.
                                Defaults to HOT's production API.
        """
        self.BASE_API_URL = base_api_url
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Referer": "geomltoolkits-python-lib",
        }

    async def request_snapshot(
        self, geometry: Dict[str, Any], feature_type: str = "building"
    ) -> Dict[str, Any]:
        """
        Request a snapshot of OSM data for a given geometry.

        Args:
            geometry (dict): GeoJSON geometry to query
            feature_type (str): Type of features to download. Defaults to "building"

        Returns:
            dict: API response containing task tracking information
        """

        payload = {
            "fileName": "geomltoolkits",
            "geometry": geometry,
            "filters": {"tags": {"all_geometry": {"join_or": {feature_type: []}}}},
            "geometryType": ["polygon"],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.BASE_API_URL}/snapshot/",
                data=json.dumps(payload),
                headers=self.headers,
            ) as response:
                response_data = await response.json()
                try:
                    response.raise_for_status()
                except Exception as ex:
                    error_message = json.dumps(response_data)
                    raise Exception(f"Error: {error_message}") from ex
                return response_data

    async def poll_task_status(self, task_link: str) -> Dict[str, Any]:
        """
        Poll the API to check the status of a submitted task.

        Args:
            task_link (str): Task tracking URL from the snapshot request

        Returns:
            dict: Task status details
        """
        async with aiohttp.ClientSession() as session:
            while True:
                async with session.get(
                    url=f"{self.BASE_API_URL}{task_link}", headers=self.headers
                ) as response:
                    response.raise_for_status()
                    res = await response.json()
                    if res["status"] in ["SUCCESS", "FAILED"]:
                        return res
                    await asyncio.sleep(2)

    async def download_snapshot(
        self,
        download_url: str,
    ) -> Dict[str, Any]:
        """
        Download the snapshot data from the provided URL.

        Args:
            download_url (str): URL to download the data
            dump (bool): Whether to save the data to a file. Defaults to False.

        Returns:
            dict: Parsed GeoJSON data
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url=download_url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.read()
                with zipfile.ZipFile(io.BytesIO(data), "r") as zip_ref:
                    with zip_ref.open("geomltoolkits.geojson") as file:
                        return json.load(file)

    async def last_updated(self) -> str:
        """
        Get the last updated date from the API status endpoint.

        Returns:
            str: The last updated date.
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.BASE_API_URL}/status", headers=self.headers
            ) as response:
                response_data = await response.json()
                try:
                    response.raise_for_status()
                except Exception as ex:
                    error_message = json.dumps(response_data)
                    raise Exception(f"Error: {error_message}") from ex
                return response_data["lastUpdated"]


async def download_osm_data(
    geojson: Optional[Union[str, dict]] = None,
    bbox: Optional[List[float]] = None,
    api_url: str = "https://api-prod.raw-data.hotosm.org/v1",
    feature_type: str = "building",
    dump: bool = False,
    out: str = None,
    split: bool = False,
    split_prefix: str = "OAM",
) -> Dict[str, Any]:
    """
    Main async function to download OSM data for a given geometry.

    Args:
        geometry (dict): GeoJSON geometry to query
        api_url (str): Base API URL
        feature_type (str): Type of features to download
        dump (bool): Whether to save the result to a file
        out (str): Output directory for saving files

    Returns:
        dict: Downloaded GeoJSON data
    """
    geometry = get_geometry(geojson, bbox)
    api = RawDataAPI(api_url)
    print("OSM Data Last Updated : ", await api.last_updated())
    task_response = await api.request_snapshot(geometry, feature_type)
    task_link = task_response.get("track_link")

    if not task_link:
        raise RuntimeError("No task link found in API response")

    result = await api.poll_task_status(task_link)

    if result["status"] == "SUCCESS" and result["result"].get("download_url"):
        download_url = result["result"]["download_url"]
        result_ret = await api.download_snapshot(download_url)

        if dump and out:
            os.makedirs(out, exist_ok=True)
            output_path = os.path.join(out, "osm-result.geojson")
            print("Dumping GeoJSON data to file...", output_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_ret, f)
            if split:
                split_dir = os.path.join(out, "split")
                print("Spllited GeoJSON wrt tiles saved to: ", split_dir)
                os.makedirs(split_dir, exist_ok=True)
                split_geojson_by_tiles(
                    output_path,
                    geojson,
                    os.path.join(out, "split"),
                    prefix=split_prefix,
                )
            return out

        return result_ret

    raise RuntimeError(f"Task failed with status: {result['status']}")


def main():
    """
    Command-line interface for OSM data download.
    """
    parser = argparse.ArgumentParser(
        description="Download GeoJSON data from the Raw Data API."
    )
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
    parser.add_argument(
        "--api-url",
        default="https://api-prod.raw-data.hotosm.org/v1",
        help="Base URL for the Raw Data API",
    )
    parser.add_argument(
        "--feature-type", default="building", help="Type of feature to download"
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.getcwd()),
        help="Directory to save downloaded tiles",
    )
    parser.add_argument(
        "--dump", action="store_true", help="Save the extracted GeoJSON data to a file"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split the output GeoJSON data into individual tiles",
    )
    args = parser.parse_args()

    async def run():
        try:
            result = await download_osm_data(
                args.aoi,
                args.bbox,
                args.api_url,
                args.feature_type,
                args.dump,
                args.out,
                args.split,
            )
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
