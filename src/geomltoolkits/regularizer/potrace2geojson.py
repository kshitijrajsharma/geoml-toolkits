#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import sys

# Additional imports for GeoPandas based geometry cleaning.
import geopandas as gpd
import numpy as np
import rasterio
from orthogonalize import orthogonalize_gdf
from PIL import Image
from shapely import make_valid
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union


def run_command(cmd):
    """
    Run a command via subprocess, logging its stdout and stderr.
    """
    logging.info("Running command: " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            logging.info("stdout:\n" + result.stdout)
        if result.stderr:
            logging.info("stderr:\n" + result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        logging.error("Command failed: " + " ".join(cmd))
        logging.error(e.stderr)
        sys.exit(1)


def convert_tif_to_bmp(input_tif, output_bmp):
    """
    Read the GeoTIFF with rasterio, then convert its first band into an 8-bit
    BMP image using Pillow. Returns the affine transform and CRS.
    """
    with rasterio.open(input_tif) as src:
        # Read the first band as a NumPy array.
        array = src.read(1)
        transform = src.transform
        crs = src.crs

    # Scale array to 0-255 if needed.
    if array.dtype != np.uint8:
        array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(
            np.uint8
        )
    array = np.flipud(array)
    # Create a PIL image and save as BMP.
    img = Image.fromarray(array)
    img.save(output_bmp, format="BMP")
    logging.info(f"BMP image saved as {output_bmp}")
    return transform, crs


def run_potrace(bmp_file, output_geojson):
    """
    Run the Potrace command:
    potrace -b geojson -o output_geojson bmp_file -i
    """
    cmd = ["potrace", "-b", "geojson", "-o", output_geojson, bmp_file, "-i"]
    run_command(cmd)


def pixel_to_geo(coord, transform):
    """
    Convert pixel coordinates (col, row) to geographic coordinates (x, y)
    using the affine transform.
    """
    # Note: coord[0] is the column (x pixel) and coord[1] is the row (y pixel)

    x, y = transform * (coord[0], coord[1])
    return [x, y]


def update_geojson_coords(geojson_file, transform, crs):
    """
    Read the GeoJSON produced by Potrace (which is in pixel coordinates),
    convert every coordinate to geographic space using the transform,
    and add CRS info.
    """
    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)

    def convert_ring(ring):
        return [pixel_to_geo(pt, transform) for pt in ring]

    for feature in geojson_data.get("features", []):
        geom = feature.get("geometry")
        if not geom:
            continue
        if geom["type"] == "Polygon":
            new_rings = []
            for ring in geom["coordinates"]:
                new_rings.append(convert_ring(ring))
            feature["geometry"]["coordinates"] = new_rings
        elif geom["type"] == "MultiPolygon":
            new_polygons = []
            for polygon in geom["coordinates"]:
                new_polygon = []
                for ring in polygon:
                    new_polygon.append(convert_ring(ring))
                new_polygons.append(new_polygon)
            feature["geometry"]["coordinates"] = new_polygons

    # Embed the CRS (non-standard but useful)
    if crs:
        geojson_data["crs"] = {
            "type": "name",
            "properties": {"name": str(crs)},
        }

    with open(geojson_file, "w") as f:
        json.dump(geojson_data, f, indent=2)
    logging.info(f"Updated GeoJSON saved as {geojson_file}")


# --- GeoPandas geometry cleaning functions ---


def linestring_to_holefree_polygon(geom):
    """
    Converts a LineString or MultiLineString into a hole-free Polygon.
    - Ensures each LineString is closed (first=last coordinate).
    - Polygonizes the result.
    - Unions them if multiple polygons result (MultiLineString).
    - Removes any holes (keeps only the exterior boundary).
    - Returns a single Polygon or MultiPolygon (hole-free).
    """
    if geom.is_empty:
        return geom

    if geom.geom_type == "LineString":
        # Ensure closed
        coords = list(geom.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)

        # Fix invalid geometry if needed
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Remove holes by creating a new polygon from the exterior
        poly = Polygon(poly.exterior)
        return poly

    elif geom.geom_type == "MultiLineString":
        # Polygonize all lines
        polys = list(polygonize(geom))
        if not polys:
            return geom  # or return None

        # Union them into one geometry
        unioned = unary_union(polys)

        # Fix invalid geometry if needed
        if not unioned.is_valid:
            unioned = unioned.buffer(0)

        # Remove holes from the unioned result
        if unioned.geom_type == "Polygon":
            return Polygon(unioned.exterior)
        elif unioned.geom_type == "MultiPolygon":
            new_polys = []
            for p in unioned.geoms:
                new_polys.append(Polygon(p.exterior))
            return unary_union(new_polys)
        else:
            # Should not happen if polygonize gave only polygons
            return unioned

    else:
        # If it's already a Polygon or MultiPolygon, etc., leave as-is or adapt as needed
        return geom


def filter_gdf_by_area(gdf, min_area=1):
    orig_crs = gdf.crs
    if orig_crs is None or orig_crs.is_geographic:
        gdf_proj = gdf.to_crs("EPSG:3857")
    else:
        gdf_proj = gdf.copy()
    gdf_proj["area_m2"] = gdf_proj.area
    gdf_proj = gdf_proj[gdf_proj["area_m2"] >= min_area].copy()
    if gdf_proj.crs != orig_crs:
        gdf_proj = gdf_proj.to_crs(orig_crs)
    return gdf_proj.drop(columns=["area_m2"])


def load_and_fix_geojson(geojson_file, simplify_tolerance=0.3):
    """
    Load the GeoJSON into a GeoDataFrame, convert any LineStrings to Polygons,
    fix invalid geometries (using buffer(0)) to cover up unwanted holes, and
    simplify with the given tolerance.
    """
    gdf = gpd.read_file(geojson_file)

    def fix_geom(geom):
        if geom is None or geom.is_empty:
            return geom

        if geom.geom_type in ["LineString", "MultiLineString"]:
            geom = linestring_to_holefree_polygon(geom)

        # If it’s a Polygon/MultiPolygon, you could also remove holes similarly:
        if geom.geom_type == "Polygon":
            # buffer(0) to fix invalid, then remove holes
            if not geom.is_valid:
                geom = geom.buffer(0)
            geom = Polygon(geom.exterior)
        elif geom.geom_type == "MultiPolygon":
            # same logic for each sub-polygon
            new_polys = []
            for p in geom.geoms:
                if not p.is_valid:
                    p = p.buffer(0)
                new_polys.append(Polygon(p.exterior))
            geom = unary_union(new_polys)

        return geom

    gdf["geometry"] = gdf["geometry"].apply(fix_geom)
    gdf["geometry"] = gdf["geometry"].simplify(
        simplify_tolerance, preserve_topology=True
    )
    return gdf


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GeoTIFF to BMP, run Potrace, update GeoJSON, and clean geometries"
    )
    parser.add_argument("input", help="Input GeoTIFF file")
    parser.add_argument("output", help="Output GeoJSON file (from Potrace)")
    parser.add_argument(
        "--tmp", default=os.getcwd(), help="Temporary filedir (default: current dir)"
    )
    parser.add_argument(
        "-n",
        "--no-orthogonalize",
        action="store_true",
        help="do not orthogonalize shape to snap 45 degrees",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Convert GeoTIFF to BMP.
    logging.info("Converting GeoTIFF to BMP...")
    transform, crs = convert_tif_to_bmp(args.input, os.path.join(args.tmp, "temp.bmp"))
    logging.info(f"Input CRS: {crs}")
    logging.info(f"Affine Transform: {transform}")

    # Run Potrace on the BMP file.
    logging.info("Running Potrace on BMP...")
    run_potrace(
        os.path.join(args.tmp, "temp.bmp"), os.path.join(args.tmp, "temp.geojson")
    )

    # Update the GeoJSON: convert pixel coordinates to geographic.
    logging.info("Updating GeoJSON coordinates...")
    update_geojson_coords(os.path.join(args.tmp, "temp.geojson"), transform, crs)

    # Load updated GeoJSON into GeoPandas, fix geometries, and simplify.
    logging.info("Loading GeoJSON into GeoPandas and cleaning geometries...")
    gdf = load_and_fix_geojson(
        os.path.join(args.tmp, "temp.geojson"), simplify_tolerance=0.2
    )
    if not args.no_orthogonalize:
        print("Orthogonalizing geometries...")
        gdf = orthogonalize_gdf(gdf)
        gdf = filter_gdf_by_area(gdf, min_area=1)
        # gdf["geometry"] = gdf["geometry"].apply(fix_invalid_polygons)
    # Write the final cleaned GeoJSON without dissolving the features.
    gdf.to_file(args.output, driver="GeoJSON")
    logging.info("Final cleaned GeoJSON written to: " + args.output)


if __name__ == "__main__":
    main()
