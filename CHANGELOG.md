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
