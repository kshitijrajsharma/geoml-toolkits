## Source : https://github.com/opengeos/geoai/blob/main/geoai/utils.py
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.affinity import rotate
from shapely.geometry import Polygon


def simple(building_polygons, angle_tolerance=10, simplify_tolerance=0.5, orthogonalize=True, preserve_topology=True):
    regularized_buildings = []
    for building in building_polygons:
        simplified = building.simplify(simplify_tolerance, preserve_topology=preserve_topology)
        if orthogonalize:
            rotated = rotate(simplified, -angle_tolerance, origin="centroid")
            ext_coords = np.array(rotated.exterior.coords)
            rect_coords = []
            for i in range(len(ext_coords) - 1):
                rect_coords.append(ext_coords[i])
                angle = (np.arctan2(ext_coords[(i + 1) % (len(ext_coords) - 1), 1] - ext_coords[i, 1], ext_coords[(i + 1) % (len(ext_coords) - 1), 0] - ext_coords[i, 0]) * 180 / np.pi)
                if abs(angle % 90) > angle_tolerance and abs(angle % 90) < (90 - angle_tolerance):
                    rect_coords.append([ext_coords[(i + 1) % (len(ext_coords) - 1), 0], ext_coords[i, 1]])
            rect_coords.append(rect_coords[0])
            regularized = Polygon(rect_coords)
            final_building = rotate(regularized, angle_tolerance, origin="centroid")
        else:
            final_building = simplified
        regularized_buildings.append(final_building)
    return gpd.GeoDataFrame(geometry=regularized_buildings, crs=building_polygons.crs)

def hybrid(building_polygons):
    results = []
    for building in building_polygons:
        complexity = building.length / (4 * np.sqrt(building.area))
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        segment_angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi
        hist, bins = np.histogram(segment_angles % 180, bins=36, range=(0, 180), weights=segment_lengths)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        dominant_angle = bin_centers[np.argmax(hist)]
        is_orthogonal = min(dominant_angle % 45, 45 - (dominant_angle % 45)) < 5
        if complexity > 1.5:
            result = building.minimum_rotated_rectangle
        elif is_orthogonal:
            rotated = rotate(building, -dominant_angle, origin="centroid")
            bounds = rotated.bounds
            ortho_hull = Polygon([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3])])
            result = rotate(ortho_hull, dominant_angle, origin="centroid")
        else:
            rotated = rotate(building, -dominant_angle, origin="centroid")
            simplified = rotated.simplify(0.3, preserve_topology=True)
            bounds = simplified.bounds
            rect_poly = Polygon([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3])])
            result = rotate(rect_poly, dominant_angle, origin="centroid")
        results.append(result)
    return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)

def adaptive(building_polygons, simplify_tolerance=0.5, area_threshold=0.9, preserve_shape=True):
    results = []
    for building in building_polygons:
        complexity = building.length / (4 * np.sqrt(building.area))
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi
        norm_angles = angles % 180
        hist, bins = np.histogram(norm_angles, bins=18, range=(0, 180), weights=segment_lengths)
        direction_clarity = np.max(hist) / np.sum(hist) if np.sum(hist) > 0 else 0
        if complexity < 1.2 and direction_clarity > 0.5:
            bin_max = np.argmax(hist)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            dominant_angle = bin_centers[bin_max]
            rotated = rotate(building, -dominant_angle, origin="centroid")
            bounds = rotated.bounds
            rect = Polygon([(bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3]), (bounds[0], bounds[3])])
            result = rotate(rect, dominant_angle, origin="centroid")
            if result.area / building.area < area_threshold or result.area / building.area > (1.0 / area_threshold):
                result = building.simplify(simplify_tolerance, preserve_topology=True)
        else:
            if preserve_shape:
                result = building.simplify(simplify_tolerance, preserve_topology=True)
            else:
                result = building.convex_hull
        results.append(result)
    return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)

def vectorize_masks(mask_path, threshold=0, min_area=10, simplify_tolerance=0.5):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        shapes = list(features.shapes(mask, transform=transform))
        polygons = []
        for shape, value in shapes:
            if value > threshold:
                polygon = Polygon(shape["coordinates"][0])
                if polygon.area >= min_area:
                    polygons.append(polygon.simplify(simplify_tolerance))
    return gpd.GeoDataFrame(geometry=polygons, crs=src.crs)

def main(mask_path):
    polygons = vectorize_masks(mask_path)
    
    regularized = simple(polygons)
    hybrid_regularized = hybrid(polygons)
    adaptive_regularized = adaptive(polygons)

    regularized.to_file("simple_regularized.geojson", driver="GeoJSON")
    hybrid_regularized.to_file("hybrid_regularized.geojson", driver="GeoJSON")
    adaptive_regularized.to_file("adaptive_regularized.geojson", driver="GeoJSON")

if __name__ == "__main__":
    mask_path = "path/to/mask.tif"
    main(mask_path)