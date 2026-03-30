# Geo ML Toolkits

Toolkits for geospatial machine learning workflows.

The project supports downloading imagery tiles and OSM features, rasterizing labels, and preparing ML datasets.

## Installation

```bash
pip install geomltoolkits
```

## Development Setup

Prerequisites:
- Python 3.10+
- uv
- just

Install dependencies:

```bash
just install
```

## Quick Python Example

```python
import os

from geomltoolkits.downloader import osm as OSMDownloader
from geomltoolkits.downloader import tms as TMSDownloader
from geomltoolkits.raster.burn import burn_labels
from geomltoolkits.raster.vectorize import vectorize_mask
from geomltoolkits.training.prepare import prepare_dataset

zoom = 18
work_dir = "banepa"
tms = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
bbox = [85.51678033745037, 27.6313353660439, 85.52323021107895, 27.637438390948745]

os.makedirs(work_dir, exist_ok=True)

chips_dir = await TMSDownloader.download_tiles(
    tms=tms,
    zoom=zoom,
    out=work_dir,
    bbox=bbox,
    georeference=True,
    dump_tile_geometries_as_geojson=True,
    prefix="OAM",
)

labels_dir = os.path.join(work_dir, "labels")
await OSMDownloader.download_osm_data(
    geojson=os.path.join(work_dir, "tiles.geojson"),
    out=labels_dir,
    dump_results=True,
)

masks_dir = os.path.join(work_dir, "masks")
burn_labels(
    labels_path=os.path.join(labels_dir, "osm-result.geojson"),
    chips_dir=chips_dir,
    output_dir=masks_dir,
)

prepare_dataset(
    chips_dir=chips_dir,
    labels_dir=masks_dir,
    output_dir=os.path.join(work_dir, "dataset"),
    dataset_format="yolo",
)

vectorize_mask(
    input_tiff=os.path.join(masks_dir, "OAM-0-0-0.tif"),
    output_geojson=os.path.join(work_dir, "vectorized.geojson"),
)
```

Detailed walkthrough notebook: [example_usage.ipynb](./example_usage.ipynb)

## CLI Commands

The package exposes a single CLI entrypoint:

```bash
geoml --help
```

Available commands:
- `geoml download-tiles`
- `geoml download-osm`
- `geoml patch`
- `geoml burn`
- `geoml prepare`

Examples:

```bash
geoml download-tiles \
  --tms "https://tiles.openaerialmap.org/.../{z}/{x}/{y}" \
  --zoom 18 \
  --bbox "85.51678033745037,27.6313353660439,85.52323021107895,27.637438390948745" \
  --out banepa \
  --georeference \
  --dump-geometries

geoml download-osm \
  --aoi banepa/tiles.geojson \
  --out banepa/labels \
  --dump

geoml burn banepa/labels/osm-result.geojson banepa/chips banepa/masks

geoml prepare banepa/chips banepa/masks banepa/dataset --dataset-format yolo
```

## Project Tasks

Use Just for local checks and development:

```bash
just lint
just typecheck
just test
just check
```

