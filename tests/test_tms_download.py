import asyncio
import os
import shutil
import tempfile
import unittest
from collections import Counter

import mercantile
from aiohttp import web
from aiohttp.test_utils import TestServer

from geomltoolkits.downloader import tms as TMSDownloader

ZOOM = 18
TILE_BYTES = b"tile"


def _bbox_of_tiles(count: int) -> list[float]:
    """Bounding box strictly inside a row of `count` tiles, so exactly those tiles are requested."""
    first = mercantile.bounds(mercantile.Tile(1000, 1000, ZOOM))
    last = mercantile.bounds(mercantile.Tile(1000 + count - 1, 1000, ZOOM))
    x_inset, y_inset = (first.east - first.west) / 10, (first.north - first.south) / 10
    return [first.west + x_inset, first.south + y_inset, last.east - x_inset, first.north - y_inset]


class TestTileDownloadReliability(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests: Counter[str] = Counter()
        self.out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out)

    async def _serve(self, handler) -> str:
        app = web.Application()
        app.router.add_get("/{z}/{x}/{y}", handler)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        return f"http://{server.host}:{server.port}/{{z}}/{{x}}/{{y}}"

    async def _download(self, tms: str, tile_count: int = 1, **options):
        return await TMSDownloader.download_tiles(
            tms=tms, zoom=ZOOM, out=self.out, bbox=_bbox_of_tiles(tile_count), max_failures=0, **options
        )

    def _saved_tiles(self) -> list[str]:
        return os.listdir(os.path.join(self.out, "chips"))

    async def test_transient_server_error_is_retried_and_the_tile_saved(self):
        async def handler(request):
            self.requests[request.path] += 1
            if self.requests[request.path] == 1:
                return web.Response(status=503)
            return web.Response(body=TILE_BYTES)

        await self._download(await self._serve(handler), max_attempts=2)

        self.assertEqual(self._saved_tiles(), [f"OAM-1000-1000-{ZOOM}.tif"])

    async def test_persistent_server_error_fails_the_download_after_all_attempts(self):
        async def handler(request):
            self.requests[request.path] += 1
            return web.Response(status=503)

        with self.assertRaisesRegex(RuntimeError, "1 > 0"):
            await self._download(await self._serve(handler), max_attempts=2)

        self.assertEqual(list(self.requests.values()), [2])

    async def test_tile_outside_the_imagery_is_skipped_without_counting_as_a_failure(self):
        async def handler(request):
            self.requests[request.path] += 1
            return web.Response(status=404)

        await self._download(await self._serve(handler), max_attempts=3)

        self.assertEqual(self._saved_tiles(), [])
        self.assertEqual(list(self.requests.values()), [1])

    async def test_client_error_is_not_retried_and_counts_as_a_failure(self):
        async def handler(request):
            self.requests[request.path] += 1
            return web.Response(status=403)

        with self.assertRaisesRegex(RuntimeError, "1 > 0"):
            await self._download(await self._serve(handler), max_attempts=3)

        self.assertEqual(list(self.requests.values()), [1])

    async def test_requests_waiting_for_a_slot_do_not_use_up_their_timeout(self):
        one_at_a_time = asyncio.Lock()

        async def handler(request):
            async with one_at_a_time:
                await asyncio.sleep(0.2)
                return web.Response(body=TILE_BYTES)

        await self._download(
            await self._serve(handler), tile_count=6, max_concurrency=1, request_timeout=0.5, max_attempts=1
        )

        self.assertEqual(len(self._saved_tiles()), 6)


if __name__ == "__main__":
    unittest.main()
