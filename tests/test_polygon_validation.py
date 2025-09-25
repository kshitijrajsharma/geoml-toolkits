#!/usr/bin/env python3

import json
import os
import tempfile
import pytest
from geomltoolkits.utils import validate_polygon_geometries, load_geojson


def create_test_geojson():
    test_features = []
    
    test_features.extend([
        {
            "type": "Feature",
            "properties": {"id": 1},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
            }
        },
        {
            "type": "Feature", 
            "properties": {"id": 2},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
            }
        },
        {
            "type": "Feature",
            "properties": {"id": 3}, 
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[4, 4], [5, 4], [5, 5], [4, 5], [4, 4]]]
            }
        }
    ])
    
    test_features.append({
        "type": "Feature",
        "properties": {"test": "invalid_polygon"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]
        }
    })
    
    test_features.append({
        "type": "Feature", 
        "properties": {"test": "point_geometry"},
        "geometry": {
            "type": "Point",
            "coordinates": [0, 0]
        }
    })
    
    test_features.append({
        "type": "Feature",
        "properties": {"test": "null_geometry"},
        "geometry": None
    })
    
    test_features.append({
        "type": "Feature",
        "properties": {"test": "multipolygon"},
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                [[[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]]]
            ]
        }
    })
    
    return {
        "type": "FeatureCollection",
        "features": test_features
    }


def test_validate_polygon_geometries_file_input():
    test_data = create_test_geojson()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
        json.dump(test_data, f)
        input_file = f.name
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as f:
            output_file = f.name
        
        result = validate_polygon_geometries(input_file, output_file)
        
        assert result == output_file
        assert os.path.exists(output_file)
        
        validated_data = load_geojson(output_file)
        assert len(validated_data["features"]) == 5
        
        for feature in validated_data["features"]:
            geom_type = feature["geometry"]["type"]
            assert geom_type in ["Polygon", "MultiPolygon"]
        
        os.unlink(output_file)
    finally:
        os.unlink(input_file)


def test_validate_polygon_geometries_dict_input():
    test_data = create_test_geojson()
    
    result = validate_polygon_geometries(test_data)
    
    assert isinstance(result, dict)
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 5
    
    for feature in result["features"]:
        geom_type = feature["geometry"]["type"]
        assert geom_type in ["Polygon", "MultiPolygon"]


def test_validate_polygon_geometries_string_input():
    test_data = create_test_geojson()
    geojson_string = json.dumps(test_data)
    
    result = validate_polygon_geometries(geojson_string)
    
    assert isinstance(result, dict)
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 5


def test_validate_polygon_geometries_empty_input():
    empty_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    with pytest.raises(ValueError, match="Empty file - no geometries provided"):
        validate_polygon_geometries(empty_data)


def test_validate_polygon_geometries_all_invalid():
    invalid_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Point",
                    "coordinates": [0, 0]
                }
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": None
            }
        ]
    }
    
    with pytest.raises(ValueError, match="No valid geometries remaining after validation"):
        validate_polygon_geometries(invalid_data)


def test_validate_complex_polygons():
    complex_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "complex_polygon"},
                "geometry": {
                    "type": "Polygon", 
                    "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]
                }
            }
        ]
    }
    
    result = validate_polygon_geometries(complex_data)
    
    assert isinstance(result, dict)
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    
    for feature in result["features"]:
        assert feature["geometry"]["type"] in ["Polygon", "MultiPolygon"]


if __name__ == "__main__":
    print("Running validation tests...")
    
    print("\n1. Testing file input/output...")
    test_validate_polygon_geometries_file_input()
    print("File input/output test passed")
    
    print("\n2. Testing dictionary input...")
    test_validate_polygon_geometries_dict_input()
    print("Dictionary input test passed")
    
    print("\n3. Testing string input...")
    test_validate_polygon_geometries_string_input()
    print("String input test passed")
    
    print("\n4. Testing empty input...")
    try:
        test_validate_polygon_geometries_empty_input()
        print("Empty input test passed")
    except ValueError:
        print("Empty input test failed - ValueError not caught properly")
    
    print("\n5. Testing all invalid input...")
    try:
        test_validate_polygon_geometries_all_invalid()
        print("All invalid input test passed")
    except ValueError:
        print("All invalid input test failed - ValueError not caught properly")
    
    print("\n6. Testing complex polygons...")
    test_validate_complex_polygons()
    print("Complex polygons test passed")
    
    print("\nAll tests completed successfully!")