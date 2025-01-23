import glob
import os
import unittest

from geomltoolkits.downloader import oam as OAMDownloader
from geomltoolkits.downloader import osm as OSMDownloader


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

    async def test_download_tiles(self):
        # Download tiles
        await OAMDownloader.download_tiles(
            tms=self.tms,
            zoom=self.zoom,
            out=self.work_dir,
            bbox=self.bbox,
            georeference=True,
            dump=True,
            prefix="OAM",
        )
        tif_files = glob.glob(os.path.join(self.work_dir, "chips", "*.tif"))
        self.assertEqual(len(tif_files), 36, "Number of .tif files should be 36")

    async def test_download_osm_data(self):
        # Download OSM data for tile boundary
        await self.test_download_tiles()

        tiles_geojson = os.path.join(self.work_dir, "tiles.geojson")
        await OSMDownloader.download_osm_data(
            geojson=tiles_geojson, out=os.path.join(self.work_dir, "labels"), dump=True
        )
        osm_result_path = os.path.join(self.work_dir, "labels", "osm-result.geojson")
        self.assertTrue(
            os.path.isfile(osm_result_path), "OSM result file should be present"
        )


if __name__ == "__main__":
    unittest.main()
