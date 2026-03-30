import collections
import functools
import os

from geopandas import GeoSeries, read_file
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid

from .._logging import get_logger, track

log = get_logger(__name__)

TOLERANCE = 1e-6
SOURCE_CRS = "EPSG:4326"
INTERMEDIATE_CRS = "EPSG:3395"

_to_intermediate = Transformer.from_crs(SOURCE_CRS, INTERMEDIATE_CRS, always_xy=True)
_to_source = Transformer.from_crs(INTERMEDIATE_CRS, SOURCE_CRS, always_xy=True)


class UndirectedGraph:
    """Simple undirected graph with DFS-based connected components."""

    def __init__(self):
        self.edges: dict[int, set[int]] = collections.defaultdict(set)

    def add_edge(self, s: int, t: int) -> None:
        self.edges[s].add(t)
        self.edges[t].add(s)

    def targets(self, v: int) -> set[int]:
        return self.edges[v]

    def vertices(self) -> collections.abc.KeysView[int]:
        return self.edges.keys()

    def dfs(self, v: int):
        stack = [v]
        seen: set[int] = set()
        while stack:
            s = stack.pop()
            if s not in seen:
                seen.add(s)
                stack.extend(self.targets(s))
                yield s

    def components(self):
        seen: set[int] = set()
        for v in self.vertices():
            if v not in seen:
                component = set(self.dfs(v))
                component.add(v)
                seen.update(component)
                yield component


def merge_polygons(
    polygons_path: str,
    output_path: str,
    distance_threshold: float,
) -> None:
    """Merge adjacent polygons within a distance threshold (meters)."""
    gdf = read_file(os.path.relpath(polygons_path))
    shapes = list(gdf["geometry"])

    graph = UndirectedGraph()
    idx = _make_index(shapes)

    for i, shape in enumerate(track(shapes, description="Building adjacency graph...")):
        embiggened = _buffer(shape, distance_threshold)
        graph.add_edge(i, i)
        nearest = [j for j in idx.intersection(embiggened.bounds, objects=False) if i != j]

        for t in nearest:
            if embiggened.intersects(shapes[t]):
                graph.add_edge(i, t)

    components = list(graph.components())
    features = []

    for component in track(components, description="Merging components..."):
        embiggened = [_buffer(shapes[v], distance_threshold) for v in component]
        merged = _unbuffer(_union(embiggened), distance_threshold)
        feature = make_valid(merged)

        if isinstance(feature, MultiPolygon):
            for polygon in feature.geoms:
                if isinstance(polygon, Polygon) and polygon.area > 0:
                    features.append(polygon)
        elif isinstance(feature, Polygon):
            features.append(feature)

    gs = GeoSeries(features).set_crs(SOURCE_CRS)
    gs.simplify(TOLERANCE).to_file(output_path, driver="GeoJSON")
    log.info(f"Merged {len(shapes)} polygons into {len(features)}, saved to {output_path}")


def _project_to_intermediate(shape):
    return shapely_transform(_to_intermediate.transform, shape)


def _project_to_source(shape):
    return shapely_transform(_to_source.transform, shape)


def _buffer(shape, distance: float):
    projected = _project_to_intermediate(shape)
    buffered = projected.buffer(distance)
    return _project_to_source(buffered)


def _unbuffer(shape, distance: float):
    projected = _project_to_intermediate(shape)
    unbuffered = projected.buffer(-1 * distance)
    return _project_to_source(unbuffered)


def _union(shapes):
    return functools.reduce(lambda lhs, rhs: lhs.union(rhs), shapes)


def _make_index(shapes):
    from rtree.index import Index, Property

    prop = Property()
    prop.dimension = 2
    prop.leaf_capacity = 1000
    prop.fill_factor = 0.9

    def bounded():
        for i, shape in enumerate(shapes):
            yield (i, shape.bounds, None)

    return Index(bounded(), properties=prop)
