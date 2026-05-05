import glob
import json
import os
import shutil
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import geopandas as gpd

from geomltoolkits.downloader import osm as OSMDownloader
from geomltoolkits.downloader import tms as TMSDownloader
from geomltoolkits.raster.vectorize import vectorize_mask


class TestDownloader(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.zoom = 18
        self.work_dir = "banepa_test"
        self.tms = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
        self.bbox = [
            85.51678033745037,
            27.6313353660439,
            85.52323021107895,
            27.637438390948745,
        ]
        os.makedirs(self.work_dir, exist_ok=True)

    async def test_download_tiles_from_tilejson(self):
        tilejson_url = "https://titiler.hotosm.org/cog/WebMercatorQuad/tilejson.json?url=https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2.tif"

        tilejson_test_dir = os.path.join(self.work_dir, "tilejson_test")
        os.makedirs(tilejson_test_dir, exist_ok=True)

        await TMSDownloader.download_tiles(
            tms=tilejson_url,
            zoom=self.zoom,
            out=tilejson_test_dir,
            bbox=self.bbox,
            georeference=True,
            dump_tile_geometries_as_geojson=True,
            prefix="TileJSON",
            is_tilejson=True,
        )

        tif_files = glob.glob(os.path.join(tilejson_test_dir, "chips", "*.tif"))
        self.assertEqual(len(tif_files), 36, "Number of .tif files should be 36")

    async def test_download_bing_tiles(self):
        bing_tms = "https://ecn.t{s}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"

        bing_test_dir = os.path.join(self.work_dir, "bing_test")
        os.makedirs(bing_test_dir, exist_ok=True)

        await TMSDownloader.download_tiles(
            tms=bing_tms,
            zoom=self.zoom,
            out=bing_test_dir,
            bbox=self.bbox,
            georeference=True,
            dump_tile_geometries_as_geojson=True,
            prefix="Bing",
        )

        tif_files = glob.glob(os.path.join(bing_test_dir, "chips", "*.tif"))
        self.assertGreater(len(tif_files), 0, "At least one .tif file should be downloaded")

        tiles_geojson = os.path.join(bing_test_dir, "tiles.geojson")
        self.assertTrue(os.path.exists(tiles_geojson))

        gdf = gpd.read_file(tiles_geojson)
        self.assertEqual(len(gdf), len(tif_files))

    async def test_download_esri_tiles(self):
        esri_tms = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}?blankTile=false"

        esri_test_dir = os.path.join(self.work_dir, "esri_test")
        os.makedirs(esri_test_dir, exist_ok=True)

        await TMSDownloader.download_tiles(
            tms=esri_tms,
            zoom=self.zoom,
            out=esri_test_dir,
            bbox=self.bbox,
            georeference=True,
            dump_tile_geometries_as_geojson=True,
            prefix="ESRI",
            tile_scheme="xyz",
        )

        tif_files = glob.glob(os.path.join(esri_test_dir, "chips", "*.tif"))
        self.assertGreater(len(tif_files), 0, "At least one .tif file should be downloaded")

        tiles_geojson = os.path.join(esri_test_dir, "tiles.geojson")
        self.assertTrue(os.path.exists(tiles_geojson))

        gdf = gpd.read_file(tiles_geojson)
        self.assertEqual(len(gdf), len(tif_files))

    async def test_download_tiles(self):
        await TMSDownloader.download_tiles(
            tms=self.tms,
            zoom=self.zoom,
            out=self.work_dir,
            bbox=self.bbox,
            georeference=True,
            dump_tile_geometries_as_geojson=True,
            prefix="OAM",
        )
        tif_files = glob.glob(os.path.join(self.work_dir, "chips", "*.tif"))
        self.assertEqual(len(tif_files), 36, "Number of .tif files should be 36")

    async def test_download_osm_data(self):
        await self.test_download_tiles()

        tiles_geojson = os.path.join(self.work_dir, "tiles.geojson")
        await OSMDownloader.download_osm_data(
            geojson=tiles_geojson,
            out=os.path.join(self.work_dir, "labels"),
            dump_results=True,
        )
        osm_result_path = os.path.join(self.work_dir, "labels", "osm-result.geojson")
        self.assertTrue(os.path.isfile(osm_result_path), "OSM result file should be present")


def _patched_session(captured: dict) -> MagicMock:
    """Build a `aiohttp.ClientSession` replacement that captures POST payloads."""

    class _RespCM:
        async def __aenter__(self_inner):
            resp = MagicMock()
            resp.json = AsyncMock(return_value={"track_link": "/tasks/abc"})
            resp.raise_for_status = MagicMock(return_value=None)
            return resp

        async def __aexit__(self_inner, *exc):
            return False

    class _SessionCM:
        async def __aenter__(self_inner):
            session = MagicMock()

            def _post(url, data, headers):
                captured["url"] = url
                captured["data"] = data
                captured["headers"] = headers
                return _RespCM()

            session.post = MagicMock(side_effect=_post)
            return session

        async def __aexit__(self_inner, *exc):
            return False

    factory = MagicMock(return_value=_SessionCM())
    return factory


class TestRawDataAPIPayload(unittest.IsolatedAsyncioTestCase):
    """Assert the JSON body sent to the raw-data API matches the public contract."""

    async def test_request_snapshot_uses_custom_filters(self):
        captured: dict = {}
        custom = {"tags": {"polygon": {"join_or": {"building": ["yes"], "amenity": ["hospital"]}}}}
        with patch("geomltoolkits.downloader.osm.aiohttp.ClientSession", _patched_session(captured)):
            api = OSMDownloader.RawDataAPI()
            await api.request_snapshot(
                geometry={"type": "Polygon", "coordinates": []},
                filters=custom,
                geometry_types=["polygon"],
            )
        body = json.loads(captured["data"])
        self.assertEqual(body["filters"], custom)
        self.assertEqual(body["geometryType"], ["polygon"])

    async def test_request_snapshot_default_filters_back_compat(self):
        captured: dict = {}
        with patch("geomltoolkits.downloader.osm.aiohttp.ClientSession", _patched_session(captured)):
            api = OSMDownloader.RawDataAPI()
            await api.request_snapshot(
                geometry={"type": "Polygon", "coordinates": []},
                feature_type="building",
            )
        body = json.loads(captured["data"])
        self.assertEqual(body["filters"], {"tags": {"all_geometry": {"join_or": {"building": []}}}})

    async def test_download_osm_data_forwards_filters(self):
        """`download_osm_data` must forward `filters` to `RawDataAPI.request_snapshot`."""
        custom = {"tags": {"polygon": {"join_or": {"amenity": ["school"]}}}}
        seen: dict = {}

        async def _fake_request_snapshot(self_, geometry, feature_type="building", geometry_types=None, filters=None):
            seen["filters"] = filters
            seen["feature_type"] = feature_type
            seen["geometry_types"] = geometry_types
            return {"track_link": "/tasks/x"}

        async def _fake_poll(self_, task_link, max_wait_seconds=600):
            return {"status": "SUCCESS", "result": {"download_url": "http://example.invalid/x.zip"}}

        async def _fake_download(self_, download_url):
            return {"type": "FeatureCollection", "features": []}

        async def _fake_last_updated(self_):
            return "2026-01-01"

        with (
            patch.object(OSMDownloader.RawDataAPI, "request_snapshot", _fake_request_snapshot),
            patch.object(OSMDownloader.RawDataAPI, "poll_task_status", _fake_poll),
            patch.object(OSMDownloader.RawDataAPI, "download_snapshot", _fake_download),
            patch.object(OSMDownloader.RawDataAPI, "last_updated", _fake_last_updated),
        ):
            result = await OSMDownloader.download_osm_data(
                geojson={"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]},
                filters=custom,
                geometry_types=["polygon"],
            )
        self.assertEqual(seen["filters"], custom)
        self.assertEqual(seen["geometry_types"], ["polygon"])
        self.assertEqual(result, {"type": "FeatureCollection", "features": []})


class TestVectorizeMasks(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_vectorize_output"
        os.makedirs(self.test_dir, exist_ok=True)
        self.input_tif = os.path.join("data", "sample_predictions.tif")
        self.output_geojson = os.path.join(self.test_dir, "sample_predictions_test.geojson")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_vectorize_masks_rasterio(self):
        if not os.path.exists(self.input_tif):
            self.skipTest(f"Input file {self.input_tif} not found.")

        vectorize_mask(
            self.input_tif,
            self.output_geojson,
            simplify_tolerance=0.2,
            min_area=1.0,
            orthogonalize=True,
        )

        self.assertTrue(os.path.exists(self.output_geojson))

        gdf_loaded = gpd.read_file(self.output_geojson)
        self.assertGreater(len(gdf_loaded), 0, "Generated GeoJSON contains no features.")


if __name__ == "__main__":
    unittest.main()
