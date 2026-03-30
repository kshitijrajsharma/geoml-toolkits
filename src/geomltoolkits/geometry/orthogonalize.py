# Adapted from Martin Machyna's orthogonalize-polygon (GPLv3)
# Source: https://github.com/Mashin6/orthogonalize-polygon

import math
import statistics

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon


def orthogonalize_gdf(
    gdf: gpd.GeoDataFrame,
    maxAngleChange: int = 45,
    skewTolerance: int = 0,
) -> gpd.GeoDataFrame:
    """Orthogonalize all geometries in a GeoDataFrame to snap to cardinal directions."""
    orig_crs = gdf.crs
    gdf_in = gdf.to_crs("EPSG:4326") if orig_crs is None or orig_crs.to_string() != "EPSG:4326" else gdf.copy()

    for i in range(len(gdf_in)):
        build = gdf_in.loc[i, "geometry"]

        if build.geom_type == "MultiPolygon":
            multipolygon = [orthogonalize_polygon(poly, maxAngleChange, skewTolerance) for poly in build.geoms]
            gdf_in.loc[i, "geometry"] = gpd.GeoSeries(MultiPolygon(multipolygon)).values
        else:
            gdf_in.loc[i, "geometry"] = orthogonalize_polygon(build, maxAngleChange, skewTolerance)

    if orig_crs is not None and orig_crs.to_string() != "EPSG:4326":
        return gdf_in.to_crs(orig_crs)
    return gdf_in


def orthogonalize_polygon(
    polygon: Polygon,
    maxAngleChange: int = 45,
    skewTolerance: int = 0,
) -> Polygon:
    """Make all angles in polygon outer and inner rings either 90 or 180 degrees."""
    rings = [Polygon(polygon.exterior)]
    for inner in polygon.interiors:
        rings.append(Polygon(inner))

    orthogonalized_rings = [_orthogonalize_ring(ring, maxAngleChange, skewTolerance) for ring in rings]

    return Polygon(
        orthogonalized_rings[0].exterior,
        [inner.exterior for inner in orthogonalized_rings[1:]],
    )


def _orthogonalize_ring(
    ring: Polygon,
    maxAngleChange: int,
    skewTolerance: int,
) -> Polygon:
    org_angle, cor_angle, dir_angle = _calculate_segment_angles(ring)
    med_angle = _compute_median_rotation(cor_angle)
    rotated = _rotate_polygon(ring, med_angle)

    org_angle, cor_angle, dir_angle = _calculate_segment_angles(rotated, maxAngleChange)
    rotated_x = rotated.exterior.xy[0].tolist()
    rotated_y = rotated.exterior.xy[1].tolist()

    shift = _find_continuation_shift(dir_angle)
    if shift != 0:
        dir_angle, org_angle, rotated_x, rotated_y = _apply_shift(dir_angle, org_angle, rotated_x, rotated_y, shift)

    dir_angle = _fix_180_turns(dir_angle)
    rotated_x, rotated_y = _average_segments(dir_angle, org_angle, rotated_x, rotated_y, skewTolerance)

    if shift != 0:
        rotated_x = rotated_x[shift:] + rotated_x[1 : shift + 1]
        rotated_y = rotated_y[shift:] + rotated_y[1 : shift + 1]
    else:
        rotated_x[0] = rotated_x[-1]
        rotated_y[0] = rotated_y[-1]

    poly_new = Polygon(zip(rotated_x, rotated_y, strict=True))
    return _rotate_polygon(poly_new, -med_angle)


def _calculate_segment_angles(
    polygon: Polygon,
    maxAngleChange: int = 45,
) -> tuple[list[float], list[float], list[int]]:
    limit_offset = 45 - maxAngleChange

    xs = polygon.exterior.xy[0]
    ys = polygon.exterior.xy[1]

    org_angle: list[float] = []
    cor_angle: list[float] = []
    dir_angle: list[int] = []
    limit = [0] * 4

    for i in range(len(xs) - 1):
        angle = _compass_bearing((ys[i], xs[i]), (ys[i + 1], xs[i + 1]))

        if angle > (45 + limit[1]) and angle <= (135 - limit[1]):
            org_angle.append(angle)
            cor_angle.append(angle - 90)
            dir_angle.append(1)
        elif angle > (135 + limit[2]) and angle <= (225 - limit[2]):
            org_angle.append(angle)
            cor_angle.append(angle - 180)
            dir_angle.append(2)
        elif angle > (225 + limit[3]) and angle <= (315 - limit[3]):
            org_angle.append(angle)
            cor_angle.append(angle - 270)
            dir_angle.append(3)
        elif angle > (315 + limit[0]) and angle <= 360:
            org_angle.append(angle)
            cor_angle.append(angle - 360)
            dir_angle.append(0)
        elif angle >= 0 and angle <= (45 - limit[0]):
            org_angle.append(angle)
            cor_angle.append(angle)
            dir_angle.append(0)

        limit = [0] * 4
        limit[dir_angle[i]] = limit_offset
        limit[(dir_angle[i] + 1) % 4] = -limit_offset
        limit[(dir_angle[i] - 1) % 4] = -limit_offset

    return org_angle, cor_angle, dir_angle


def _compass_bearing(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    if not isinstance(point_a, tuple) or not isinstance(point_b, tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(point_a[0])
    lat2 = math.radians(point_b[0])
    diff_long = math.radians(point_b[1] - point_a[1])

    x = math.sin(diff_long) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_long)

    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _rotate_polygon(polygon: Polygon, angle: float) -> Polygon:
    gs = gpd.GeoDataFrame({"geometry": [polygon]})
    gs.crs = "EPSG:4326"
    rotated = gs.to_crs("EPSG:3857")
    rotated = rotated.rotate(angle, origin="centroid", use_radians=False)
    rotated = rotated.to_crs("EPSG:4326")
    return rotated.iloc[0]


def _compute_median_rotation(cor_angle: list[float]) -> float:
    if statistics.stdev(cor_angle) < 30:
        return statistics.median(cor_angle)
    return 45


def _find_continuation_shift(dir_angle: list[int]) -> int:
    shift = 0
    for i in range(1, len(dir_angle)):
        if dir_angle[0] == dir_angle[-i]:
            shift = i
        else:
            break
    return shift


def _apply_shift(
    dir_angle: list[int],
    org_angle: list[float],
    rotated_x: list[float],
    rotated_y: list[float],
    shift: int,
) -> tuple[list[int], list[float], list[float], list[float]]:
    dir_angle = dir_angle[-shift:] + dir_angle[:-shift]
    org_angle = org_angle[-shift:] + org_angle[:-shift]
    rotated_x = rotated_x[-shift - 1 : -1] + rotated_x[:-shift]
    rotated_y = rotated_y[-shift - 1 : -1] + rotated_y[:-shift]
    return dir_angle, org_angle, rotated_x, rotated_y


def _fix_180_turns(dir_angle: list[int]) -> list[int]:
    dir_angle_roll = dir_angle[1:] + dir_angle[0:1]
    return [
        dir_angle[i - 1] if abs(dir_angle[i] - dir_angle_roll[i]) == 2 else dir_angle[i] for i in range(len(dir_angle))
    ]


def _average_segments(
    dir_angle: list[int],
    org_angle: list[float],
    rotated_x: list[float],
    rotated_y: list[float],
    skew_tolerance: int,
) -> tuple[list[float], list[float]]:
    dir_angle = [*dir_angle, dir_angle[0]]
    org_angle = [*org_angle, org_angle[0]]
    segment_buffer: list[int] = []

    for i in range(len(dir_angle) - 1):
        if org_angle[i] % 90 > (45 - skew_tolerance) and org_angle[i] % 90 < (45 + skew_tolerance):
            continue

        segment_buffer.append(i)
        if dir_angle[i] == dir_angle[i + 1]:
            next_skewed = org_angle[i + 1] % 90 > (45 - skew_tolerance) and org_angle[i + 1] % 90 < (
                45 + skew_tolerance
            )
            if not next_skewed:
                continue

        if dir_angle[i] in {0, 2}:
            mean_x = statistics.mean(rotated_x[segment_buffer[0] : segment_buffer[-1] + 2])
            rotated_x[segment_buffer[0] : segment_buffer[-1] + 2] = [mean_x] * (len(segment_buffer) + 1)
        elif dir_angle[i] in {1, 3}:
            mean_y = statistics.mean(rotated_y[segment_buffer[0] : segment_buffer[-1] + 2])
            rotated_y[segment_buffer[0] : segment_buffer[-1] + 2] = [mean_y] * (len(segment_buffer) + 1)

        if 0 in segment_buffer:
            rotated_x[-1] = rotated_x[0]
            rotated_y[-1] = rotated_y[0]

        segment_buffer = []

    return rotated_x, rotated_y
