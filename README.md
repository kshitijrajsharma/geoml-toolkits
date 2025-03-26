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

### Detailed Usage Instructions

#### Downloading Tiles

To download tiles from a Tile Map Service (TMS), you can use the `download_tiles` function from the `tms` module. Here is an example:

```python
import os
from geomltoolkits.downloader import tms as TMSDownloader

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
```

#### Downloading OSM Data

To download OpenStreetMap (OSM) data for a given area of interest, you can use the `download_osm_data` function from the `osm` module. Here is an example:

```python
import os
from geomltoolkits.downloader import osm as OSMDownloader

# Define area of interest
WORK_DIR = "banepa"
tiles_geojson = os.path.join(WORK_DIR, "tiles.geojson")

# Download OSM data for tile boundary
await OSMDownloader.download_osm_data(
    geojson=tiles_geojson,
    out=os.path.join(WORK_DIR, "labels"),
    dump=True,
    split=True
)
```

#### Regularizing Footprints

This repo utiltizes a digital art technique to vectorize features from masks using our awesome old [potrace](https://potrace.sourceforge.net/) library . Potrace is not meant for geospatial workflows however it does excellent job on tracing vector graphics from raster. I  thought it would easily overcome the current rasteriation issues of irregular geometries and hence worked on the spatial integration . Below is the example how potrace works 
![image](https://github.com/user-attachments/assets/e2f41ff2-9226-4f53-be84-4ae22cb43360)

Repo uses orthogonalization script and potrace also provides rasterio rasterization option as well ! 

Example of vectorization output : 

![image](https://github.com/user-attachments/assets/ff880321-6ff3-46b0-b098-9426c3a31fc6)



To regularize building footprints, you can use the `VectorizeMasks` class from the `regularizer` module. Here is an example:

```python
import os
from geomltoolkits.regularizer import VectorizeMasks

# Define input and output files
input_tiff = "path/to/input.tiff"
output_geojson = "path/to/output.geojson"

# Create a VectorizeMasks instance
converter = VectorizeMasks(
    simplify_tolerance=0.2,
    min_area=1.0,
    orthogonalize=True,
    algorithm="potrace",
    tmp_dir=os.getcwd()
)

# Run the conversion
converter.convert(input_tiff, output_geojson)
```

#### Command Line Usage

The GeoML Toolkits also provide command line interfaces for downloading tiles, downloading OSM data, and regularizing footprints. Here are the commands:

- **tmd**: TMS downloader
- **osd**: OpenStreetMap downloader
- **reg**: Footprints regularizer

You can use the `--help` option with each command to see the available options and usage instructions. For example:

```sh
tmd --help
osd --help
reg --help
```

#### Example Usage

For a complete example of how to use the GeoML Toolkits, you can refer to the `example_usage.ipynb` notebook. It provides detailed explanations and usage examples for all functionalities.

#### Installation Steps

To install the GeoML Toolkits, you can use pip:

```sh
pip install geomltoolkits
```

If you want to install the library locally for development, you can use the following commands:

```sh
pip install poetry
poetry install
pip install -e .
```

#### Running the Bash Script

You can also use the provided Bash script to run the GeoML Toolkits from the command line. Here is an example:

```sh
./run.sh
```

The script will download tiles, download OSM data, and regularize footprints based on the specified parameters.

