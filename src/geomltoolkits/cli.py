import asyncio
from typing import Annotated

import typer

app = typer.Typer(name="geoml", help="Geospatial Machine Learning Toolkits CLI")


@app.command()
def download_tiles(
    tms: Annotated[str, typer.Option(help="TMS URL template")],
    zoom: Annotated[int, typer.Option(help="Zoom level")],
    aoi: Annotated[str | None, typer.Option(help="GeoJSON file path or string")] = None,
    bbox: Annotated[str | None, typer.Option(help="Bounding box: xmin,ymin,xmax,ymax")] = None,
    out: Annotated[str, typer.Option(help="Output directory")] = ".",
    georeference: Annotated[bool, typer.Option(help="Georeference downloaded tiles")] = False,
    crs: Annotated[str, typer.Option(help="CRS for georeferenced tiles")] = "4326",
    prefix: Annotated[str, typer.Option(help="Filename prefix")] = "OAM",
    scheme: Annotated[str | None, typer.Option(help="Tile URL scheme")] = None,
    tilejson: Annotated[bool, typer.Option(help="Treat TMS URL as TileJSON")] = False,
    dump_geometries: Annotated[bool, typer.Option(help="Dump tile geometries as GeoJSON")] = False,
) -> None:
    """Download tiles from a TMS source for a given area."""
    from .downloader.tms import download_tiles as _download

    bbox_list = [float(x) for x in bbox.split(",")] if bbox else None

    asyncio.run(
        _download(
            tms=tms,
            zoom=zoom,
            out=out,
            geojson=aoi,
            bbox=bbox_list,
            georeference=georeference,
            dump_tile_geometries_as_geojson=dump_geometries,
            prefix=prefix,
            crs=crs,
            tile_scheme=scheme,
            is_tilejson=tilejson,
        )
    )


@app.command()
def download_osm(
    aoi: Annotated[str | None, typer.Option(help="GeoJSON file path or string")] = None,
    bbox: Annotated[str | None, typer.Option(help="Bounding box: xmin,ymin,xmax,ymax")] = None,
    feature_type: Annotated[str, typer.Option(help="OSM feature type")] = "building",
    out: Annotated[str, typer.Option(help="Output directory")] = ".",
    crs: Annotated[str, typer.Option(help="Output CRS")] = "4326",
    dump: Annotated[bool, typer.Option(help="Save results to file")] = True,
) -> None:
    """Download OSM data via the HOT Raw Data API."""
    from .downloader.osm import download_osm_data

    bbox_list = [float(x) for x in bbox.split(",")] if bbox else None

    asyncio.run(
        download_osm_data(
            geojson=aoi,
            bbox=bbox_list,
            feature_type=feature_type,
            dump_results=dump,
            out=out,
            crs=crs,
        )
    )


@app.command()
def patch(
    input_tiff: Annotated[str, typer.Argument(help="Input COG/GeoTIFF")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for patches")],
    patch_size: Annotated[int, typer.Option(help="Patch size in pixels")] = 256,
    prefix: Annotated[str, typer.Option(help="Filename prefix")] = "patch",
) -> None:
    """Create georeferenced patches (chips) from a COG/GeoTIFF."""
    from .raster.patch import create_patches

    create_patches(input_tiff, output_dir, patch_size=patch_size, prefix=prefix)


@app.command()
def burn(
    labels: Annotated[str, typer.Argument(help="GeoJSON labels file")],
    chips_dir: Annotated[str, typer.Argument(help="Directory of chip GeoTIFFs")],
    output_dir: Annotated[str, typer.Argument(help="Output directory for masks")],
    burn_value: Annotated[int, typer.Option(help="Burn value for binary masks")] = 255,
    class_property: Annotated[str | None, typer.Option(help="Feature property for multi-class")] = None,
) -> None:
    """Rasterize GeoJSON labels onto matching chips as masks."""
    from .raster.burn import burn_labels

    burn_labels(labels, chips_dir, output_dir, burn_value=burn_value, class_property=class_property)


@app.command()
def prepare(
    chips_dir: Annotated[str, typer.Argument(help="Directory of chip images")],
    labels_dir: Annotated[str, typer.Argument(help="Directory of label masks")],
    output_dir: Annotated[str, typer.Argument(help="Output dataset directory")],
    dataset_format: Annotated[str, typer.Option(help="Dataset format: yolo or coco")] = "yolo",
) -> None:
    """Prepare a training dataset from chips and labels."""
    from .training.prepare import prepare_dataset

    prepare_dataset(chips_dir, labels_dir, output_dir, dataset_format=dataset_format)
