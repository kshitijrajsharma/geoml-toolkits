#!/bin/bash

# Print script header
echo "===== Running GeoML Toolkits ====="

# Default values
zoom=18
work_dir_name="banepa"
tms="https://tiles.openaerialmap.org/64d3642319cb3a000147a5be/0/64d3642319cb3a000147a5bf/{z}/{x}/{y}"
aoi="input.geojson"

# Get the current working directory
pwd=$(pwd)

# Construct the work_dir path
work_dir="$pwd/$work_dir_name"

# Check if arguments are provided
if [ "$#" -gt 0 ]; then
    zoom=$1
    echo "Using zoom level: $zoom"
else
    echo "Using default zoom level: $zoom"
fi

if [ "$#" -gt 1 ]; then
    work_dir_name=$2
    work_dir="$pwd/$work_dir_name"
    echo "Using work directory: $work_dir"
else
    echo "Using default work directory: $work_dir"
fi

if [ "$#" -gt 2 ]; then
    tms=$3
    echo "Using TMS: $tms"
else
    echo "Using default TMS: $tms"
fi

if [ "$#" -gt 3 ]; then
    aoi=$4
    echo "Using AOI: $aoi"
else
    echo "Using default AOI: $aoi"
fi

# Create the work directory if it doesn't exist
echo "Creating work directory: $work_dir"
mkdir -p "$work_dir"

# Download images first as chips
echo "Downloading images as chips..."
tmd --aoi $aoi --tms $tms --zoom $zoom --out "$work_dir" --georeference --dump

# Download osm data in the tile boundary
echo "Downloading OSM data in the tile boundary..."
osd --aoi "$work_dir/tiles.geojson" --dump --out "$work_dir"

# Regularize footprints
echo "Regularizing footprints..."
reg --input "$work_dir/tiles.geojson" --output "$work_dir/regularized.geojson"

echo "===== Script completed successfully ====="
