### Geo ML Toolkits 

Toolkits for GeoML workflows 

Currently it supports for downloading and processing geospatial data from Open Aerial Map (OAM) and OpenStreetMap (OSM). This toolkit allows you to define an area of interest, download aerial imagery and OSM data, and generate training data for machine learning models.

## Installation

To install the GeoML Toolkits, you can use pip:

```sh
pip install geomltoolkits
```

## Usage
### Python Example 

Below is an example of how to use the GeoML Toolkits to download and process geospatial data.

```python
import os
from geomltoolkits.downloader import tms as TMSDownloader
from geomltoolkits.downloader import osm as OSMDownloader

# Define area of interest
ZOOM = 18
WORK_DIR = "banepa"
TMS = "https://tiles.openaerialmap.org/62d85d11d8499800053796c1/0/62d85d11d8499800053796c2/{z}/{x}/{y}"
BBOX = [85.514668, 27.628367, 85.528875, 27.638514]

# Create working directory
os.makedirs(WORK_DIR, exist_ok=True)

# Download tiles
await TMSDownloader.download_tiles(
    tms=TMS,
    zoom=ZOOM,
    out=WORK_DIR,
    bbox=BBOX,
    georeference=True,
    dump=True,
    prefix="OAM"
)

# Download OSM data for tile boundary
tiles_geojson = os.path.join(WORK_DIR, "tiles.geojson")
await OSMDownloader.download_osm_data(
    geojson=tiles_geojson,
    out=os.path.join(WORK_DIR, "labels"),
    dump=True,
    split=True
)
```
Learn more [here](./example_usage.ipynb) 

### Command Line Usage
if you install the python package it will by default install following  commands 

- **tmd**: tms downloader
- **osd** : openstreetmap downloader
- **reg** : footprints regularizer


You can see the helper function and shoot your command
   
You can also use the provided Bash script to run the GeoML Toolkits from the command line. 
Take a look [here](./run.sh)

#### Splitting OSM Data

If the split argument is set to True, the downloaded OSM data will be split based on the tiles defined in tiles.geojson. Each resulting GeoJSON file will be named according to the tile's x, y, and z values.

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- [poetry](https://python-poetry.org/) 

### Install Poetry

If you don't have poetry installed, you can install it using the following command:

```bash
pip install poetry
```

#### Install 
```bash
poetry install
```

### Install lib locally 
```bash
pip install -e . 
``` 

