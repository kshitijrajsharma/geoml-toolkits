## 0.3.6 (2025-05-24)

### Fix

- **extension**: fixes png not being available to the extension

## 0.3.5 (2025-05-22)

### Fix

- **tqdm**: bump tqdm version

## 0.3.4 (2025-05-22)

### Fix

- **compatibility**: adds pillow compability on the pillow

## 0.3.3 (2025-05-21)

### Fix

- **python**: bumps python version
- **workflow**: update uv setup step to use version 6 and streamline Python installation

## 0.3.2 (2025-05-21)

### Fix

- **shapely-version**: downgrade shapely , support smaller version

## 0.3.1 (2025-05-21)

### Fix

- **shapely**: support old version of shapely
- **workflow**: remove redundant group flags in dependency sync step

## 0.3.0 (2025-05-12)

### Feat

- **uv-support**: removes poetry and adds uv

### Fix

- **workflow**: adjust job strategy and permissions in test workflow

## 0.2.0 (2025-05-07)

### Feat

- **tile**: enhance tile fetching and URL handling; support TileJSON and improved georeferencing

### Fix

- **tms**: adds multiple tms support and fixes for the negative xyz server to download

## 0.1.6 (2025-03-25)

### Fix

- **size**: fixes filesize to be hardcoded in tiles

## 0.1.5 (2025-03-25)

### Fix

- **refact**: debug statement

## 0.1.4 (2025-03-25)

### Fix

- **dir**: fix new dir if there is in tmp

## 0.1.3 (2025-03-25)

### Fix

- **typo**: fixes mistakely removen function

## 0.1.2 (2025-03-25)

### Fix

- **edge**: fixes edge artifacts during georeferencing

## 0.1.1 (2025-03-24)

### Fix

- **georef**: adds tile georeferencing
- **consistency**: adds consistency on the crs
- **crs**: adds support for 3857
- **cmd**: update description for GeoTIFF to GeoJSON conversion

## 0.1.0 (2025-03-24)

### Feat

- **vectorizer**: adds vectorizer using potrace and orthogonalization
- **regularizer**: import VectorizeMasks from app module
- **regularizer**: add area filtering and optional orthogonalization to GeoJSON processing
- **regularizer**: add script to convert GeoTIFF to BMP and update GeoJSON
- **regularizer**: add SVG to GeoJSON conversion script
- **utils**: add function to merge raster files into a single output
- **downloader**: implement TMS downloader and update usage examples

### Refactor

- **ci**: adds pillow
- **regularizer**: move orthogonalize_gdf import to app module
- **regularizer**: clean up logging configuration and fix geometry type check
- **regularizer**: remove unused SVG and Potrace conversion scripts

## 0.0.2 (2025-01-23)

### Fix

- **osm**: added split feature on labels
