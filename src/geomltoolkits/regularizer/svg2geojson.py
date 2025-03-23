#!/usr/bin/env python
import json
import sys
import xml.etree.ElementTree as ET

from osgeo import gdal
from svgpathtools import parse_path


def extract_coords_from_path_d(d, samples_per_segment=20):
    """
    Parses an SVG 'd' attribute using svgpathtools and
    returns a list of (x, y) coordinates by sampling each segment.
    """
    path = parse_path(d)
    coords = []
    for segment in path:
        for i in range(samples_per_segment):
            t = i / (samples_per_segment - 1)
            pt = segment.point(t)
            coords.append((pt.real, pt.imag))
    return coords


def apply_geotransform(x, y, gt):
    """
    Applies the affine geotransform from the TIFF to a coordinate (x,y).

    Geotransform gt is a tuple with:
      gt[0] = top left x, gt[1] = pixel width, gt[2] = rotation (x)
      gt[3] = top left y, gt[4] = rotation (y), gt[5] = pixel height (typically negative)
    """
    X_geo = gt[0] + x * gt[1] + y * gt[2]
    Y_geo = gt[3] + x * gt[4] + y * gt[5]
    return (X_geo, Y_geo)


def main():

    tiff_file = "merged4326.tif"
    svg_file = "test.svg"
    output_file = "potrace.geojson"

    ds = gdal.Open(tiff_file)
    if ds is None:
        print("Error: Could not open TIFF file:", tiff_file)
        sys.exit(1)

    gt = ds.GetGeoTransform()
    print("TIFF GeoTransform parameters:", gt)

    try:
        tree = ET.parse(svg_file)
    except Exception as e:
        print("Error parsing SVG:", e)
        sys.exit(1)

    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    features = []

    for path_elem in root.findall(".//svg:path", ns):
        d = path_elem.get("d")
        if not d:
            continue

        coords = extract_coords_from_path_d(d)
        if not coords:
            continue

        geo_coords = [apply_geotransform(x, y, gt) for (x, y) in coords]

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": geo_coords},
            "properties": {},
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    try:
        with open(output_file, "w") as f:
            json.dump(geojson, f, indent=2)
        print("GeoJSON successfully written to", output_file)
    except Exception as e:
        print("Error writing GeoJSON:", e)


if __name__ == "__main__":
    main()
