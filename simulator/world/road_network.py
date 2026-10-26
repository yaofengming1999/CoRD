from __future__ import annotations
import math
import pickle
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from ..model.entities import Location
from ..utils.geo import haversine_distance

logger = logging.getLogger(__name__)

_OSMNX_AVAILABLE = False
try:
    import osmnx as ox
    import networkx as nx
    _OSMNX_AVAILABLE = True
except ImportError:
    logger.warning("osmnx not installed — nearest-node queries will use brute-force graph search.")


class RoadNetwork:
    """Wrapper around an OSM road graph.

    In strict mode, routing failures are surfaced as errors instead of being
    hidden by straight-line fallbacks.
    """

    # Approximate factor to convert straight-line to road distance
    _DEFAULT_DETOUR = 1.3

    def __init__(self, config: dict):
        self.config = config
        self._detour = config.get("map", {}).get("road_detour_factor", self._DEFAULT_DETOUR)
        self._place = config.get("map", {}).get("place_name", "")
        self._strict_nearest_node = config.get("map", {}).get("strict_nearest_node", True)
        # Snap-distance guard: a location that snaps farther than this from
        # its nearest graph node means the coordinates and the loaded graph
        # disagree (e.g. Shanghai instances routed on the Exeter map — every
        # point collapsed to one corner node and all orders completed
        # instantly). In-map snaps are typically <300 m, so 2 km separates
        # "sparse road digitization" from "wrong map" by orders of magnitude.
        self._max_snap_m = float(config.get("map", {}).get("max_snap_distance_m", 2000.0))
        self._repo_root = Path(__file__).resolve().parent.parent.parent
        self._map_dir = self._resolve_map_dir(config)
        self.graph = None          # drive graph
        self.walk_graph = None     # walk / pedestrian graph
        self._bounds: Optional[dict] = None
        # Largest strongly-connected component per mode. OSM extracts contain
        # small disconnected islands (one-way systems, service roads); a
        # location snapped to an island node has no path to the rest of the
        # network in either direction. All snapping and routing is therefore
        # canonicalized to the main component. Nodes are NOT removed from the
        # graph so instance-stored node ids remain resolvable.
        self._main_nodes: dict[str, set] = {}
        # Exact-key memo caches. The map is static, so a (mode, node, node)
        # Dijkstra result and a (mode, lat, lon) nearest-node snap never go
        # stale; keys are exact floats/ids so cached answers are bit-identical
        # to recomputation. Size caps only bound memory on very long runs.
        self._nearest_cache: dict[tuple, int] = {}
        self._dijkstra_cache: "OrderedDict[tuple, float]" = OrderedDict()
        # Full single-source distance maps. One SSSP costs ~2x a targeted
        # query but serves every destination from that source; insertion
        # sweeps query many destinations per moving origin, so
        # this collapses the per-epoch Dijkstra count by ~10x. Values are
        # produced by the same Dijkstra weights, so results stay bit-identical.
        #
        # LRU eviction, NOT clear-all: the distinct-source working set of a
        # matching episode (origins + queued/candidate stops snapped to
        # nodes) is ~800 on Exeter and is stable across episodes, so a cap
        # above it lets the cache warm once and then serve ~100% hits for the
        # rest of training. The old 512 clear-all cap sat *below* that working
        # set, so it wiped mid-episode and every episode re-paid the full
        # Dijkstra sweep — the dominant cost once insertion-aware admission /
        # projection features are on (road Dijkstra was ~45% of runtime).
        self._sssp_cache: "OrderedDict[tuple, dict]" = OrderedDict()
        self._NEAREST_CACHE_MAX = 500_000
        self._DIJKSTRA_CACHE_MAX = 2_000_000
        self._SSSP_CACHE_MAX = 4096

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self):
        self.graph = self._load_graph_file(self._graph_path("drive"), "drive")
        self.walk_graph = self._load_graph_file(self._graph_path("walk"), "walk")
        self._main_nodes["drive"] = self._largest_component_nodes(self.graph, "drive")
        self._main_nodes["walk"] = self._largest_component_nodes(self.walk_graph, "walk")
        self._build_node_arrays()

        self._compute_bounds()

    def _build_node_arrays(self) -> None:
        """Precompute per-mode node coordinate arrays for nearest-node queries.

        ox.distance.nearest_nodes rebuilds a GeoDataFrame of every node on
        each call (~11 ms); a vectorized haversine argmin over these arrays
        answers the same query in ~0.1 ms with the same great-circle metric.
        """
        self._node_arrays = {}
        for mode, g in (("drive", self.graph), ("walk", self.walk_graph)):
            if g is None:
                continue
            ids, lats, lons = [], [], []
            for node_id, data in g.nodes(data=True):
                if "y" not in data or "x" not in data:
                    continue
                ids.append(int(node_id))
                lats.append(float(data["y"]))
                lons.append(float(data["x"]))
            if not ids:
                continue
            id_arr = np.asarray(ids, dtype=np.int64)
            main = self._main_nodes.get(mode) or set()
            self._node_arrays[mode] = (
                id_arr,
                np.radians(np.asarray(lats)),
                np.radians(np.asarray(lons)),
                np.fromiter((i in main for i in ids), dtype=bool, count=len(ids)),
            )

    def _vector_nearest(self, location: Location, mode: str, *, main_only: bool) -> Optional[int]:
        arrays = getattr(self, "_node_arrays", {}).get(mode)
        if arrays is None:
            return None
        id_arr, lats, lons, main_mask = arrays
        lat0 = math.radians(location.lat)
        lon0 = math.radians(location.lon)
        # Haversine "a" term is monotone in great-circle distance, so its
        # argmin identifies the same nearest node without asin/radius.
        a = np.sin((lats - lat0) / 2.0) ** 2 + math.cos(lat0) * np.cos(lats) * np.sin((lons - lon0) / 2.0) ** 2
        if main_only:
            if not main_mask.any():
                return None
            a = np.where(main_mask, a, np.inf)
        return int(id_arr[int(np.argmin(a))])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def nearest_node(self, location: Location, mode: str = "drive") -> int:
        cache_key = (mode, location.lat, location.lon)
        cached = self._nearest_cache.get(cache_key)
        if cached is not None:
            return cached
        node = self._nearest_node_uncached(location, mode)
        self._check_snap_distance(location, node, mode)
        if len(self._nearest_cache) >= self._NEAREST_CACHE_MAX:
            self._nearest_cache.clear()
        self._nearest_cache[cache_key] = node
        return node

    def _check_snap_distance(self, location: Location, node: int, mode: str) -> None:
        if node == -1 or self._max_snap_m <= 0:
            return
        g = self._select_graph(mode)
        if g is None or node not in g.nodes:
            return
        data = g.nodes[node]
        if "y" not in data or "x" not in data:
            return
        snap_m = haversine_distance(location.lat, location.lon, float(data["y"]), float(data["x"]))
        if snap_m <= self._max_snap_m:
            return
        message = (
            f"Location lat={location.lat}, lon={location.lon} snapped {snap_m/1000:.1f} km away "
            f"to node {node} (mode={mode}, map={self._place!r}) — the coordinates and the loaded "
            "road graph almost certainly belong to different cities."
        )
        if self._strict_nearest_node:
            raise RuntimeError(message)
        logger.warning("%s", message)

    def _nearest_node_uncached(self, location: Location, mode: str) -> int:
        g = self._select_graph(mode)
        if g is None:
            if self._strict_nearest_node:
                raise RuntimeError(f"Road graph is not loaded for mode={mode}")
            return -1
        node = self._vector_nearest(location, mode, main_only=False)
        if node is not None:
            main = self._main_nodes.get(mode)
            if main and node not in main:
                # Snapped onto a disconnected island; re-snap within the
                # routable main component (same two-stage logic as before).
                resnapped = self._vector_nearest(location, mode, main_only=True)
                if resnapped is not None:
                    return resnapped
            else:
                return node
        if not _OSMNX_AVAILABLE:
            return self._nearest_node_bruteforce(g, location, allowed=self._main_nodes.get(mode))
        try:
            node, _ = ox.distance.nearest_nodes(g, location.lon, location.lat, return_dist=True)
            main = self._main_nodes.get(mode)
            if main and int(node) not in main:
                # Snapped onto a disconnected island; re-snap within the
                # routable main component instead.
                return self._nearest_node_bruteforce(g, location, allowed=main)
            return int(node)
        except Exception as exc:
            if self._strict_nearest_node:
                raise RuntimeError(
                    f"Failed to find nearest OSM node for "
                    f"lat={location.lat}, lon={location.lon}, mode={mode}"
                ) from exc
            logger.warning(
                "OSMnx nearest_nodes failed for lat=%s lon=%s mode=%s; using brute-force fallback: %s",
                location.lat,
                location.lon,
                mode,
                exc,
            )
            return self._nearest_node_bruteforce(g, location, allowed=self._main_nodes.get(mode))

    def node_location(self, node_id: int, mode: str = "drive") -> Optional[Location]:
        g = self._select_graph(mode)
        if g is None or node_id not in g.nodes:
            return None
        data = g.nodes[node_id]
        return Location(lat=data["y"], lon=data["x"], node_id=node_id)

    def shortest_path_distance(
        self,
        origin: Location,
        destination: Location,
        mode: str = "drive",
    ) -> float:
        """Return road distance in metres."""
        g = self._select_graph(mode)
        if g is None:
            if self._strict_nearest_node:
                raise RuntimeError(f"Road graph is not loaded for mode={mode}")
            return haversine_distance(origin.lat, origin.lon, destination.lat, destination.lon) * self._detour

        try:
            o_node = self._routable_node(origin, mode)
            d_node = self._routable_node(destination, mode)
            cache_key = (mode, o_node, d_node)
            cached = self._dijkstra_cache.get(cache_key)
            if cached is not None:
                self._dijkstra_cache.move_to_end(cache_key)
                return cached
            import networkx as nx
            sssp_key = (mode, o_node)
            dist_map = self._sssp_cache.get(sssp_key)
            if dist_map is None:
                dist_map = nx.single_source_dijkstra_path_length(g, o_node, weight="length")
                if len(self._sssp_cache) >= self._SSSP_CACHE_MAX:
                    self._sssp_cache.popitem(last=False)
                self._sssp_cache[sssp_key] = dist_map
            else:
                self._sssp_cache.move_to_end(sssp_key)
            length = dist_map.get(d_node)
            if length is None:
                # Destination unreachable forward (cannot happen once both
                # nodes sit in the main SCC): same reverse fallback as the
                # old has_path branch.
                length = nx.shortest_path_length(g, d_node, o_node, weight="length")
            result = float(length)
            if len(self._dijkstra_cache) >= self._DIJKSTRA_CACHE_MAX:
                self._dijkstra_cache.popitem(last=False)
            self._dijkstra_cache[cache_key] = result
            return result
        except Exception as exc:
            if self._strict_nearest_node:
                raise RuntimeError(
                    "Failed to compute road shortest-path distance "
                    f"from lat={origin.lat}, lon={origin.lon} "
                    f"to lat={destination.lat}, lon={destination.lon}, mode={mode}"
                ) from exc
            return haversine_distance(origin.lat, origin.lon, destination.lat, destination.lon) * self._detour

    def shortest_path(
        self,
        origin: Location,
        destination: Location,
        mode: str = "drive",
    ) -> list[int]:
        """Return list of node IDs on the shortest path."""
        g = self._select_graph(mode)
        if g is None:
            if self._strict_nearest_node:
                raise RuntimeError(f"Road graph is not loaded for mode={mode}")
            return []
        import networkx as nx
        o_node = self._routable_node(origin, mode)
        d_node = self._routable_node(destination, mode)
        try:
            if nx.has_path(g, o_node, d_node):
                shortest_path = nx.shortest_path(g, o_node, d_node, weight="length")
            else:
                shortest_path = nx.shortest_path(g, d_node, o_node, weight="length")
        except nx.NetworkXNoPath:
            # Both directions unreachable even after canonicalization (should
            # not happen once endpoints sit in the main component); degrade to
            # a straight two-node leg rather than killing the episode.
            logger.warning(
                "No road path between nodes %s and %s (mode=%s); using direct leg.",
                o_node, d_node, mode,
            )
            return [int(o_node), int(d_node)]
        return [int(n) for n in shortest_path]

    def sample_random_node(self, rng, mode: str = "drive") -> Optional[Location]:
        """Sample a uniformly random node from the main routable component."""
        g = self._select_graph(mode)
        if g is None:
            return None
        main = self._main_nodes.get(mode)
        nodes = [(nid, data) for nid, data in g.nodes(data=True) if not main or nid in main]
        if not nodes:
            return None
        idx = int(rng.integers(len(nodes)))
        nid, data = nodes[idx]
        return Location(lat=data["y"], lon=data["x"], node_id=int(nid))

    def bounds(self) -> Optional[dict]:
        return self._bounds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_graph(self, mode: str):
        if mode in ("walk", "pedestrian"):
            return self.walk_graph
        return self.graph

    def _nearest_node_bruteforce(self, graph, location: Location, allowed: Optional[set] = None) -> int:
        best_node = None
        best_distance = float("inf")
        for node_id, data in graph.nodes(data=True):
            if "y" not in data or "x" not in data:
                continue
            if allowed is not None and node_id not in allowed:
                continue
            distance = haversine_distance(location.lat, location.lon, data["y"], data["x"])
            if distance < best_distance:
                best_node = node_id
                best_distance = distance
        return int(best_node) if best_node is not None else -1

    def _largest_component_nodes(self, graph, mode: str) -> set:
        """Node set of the largest (strongly) connected component."""
        if graph is None or not _OSMNX_AVAILABLE:
            return set()
        if graph.is_directed():
            components = nx.strongly_connected_components(graph)
        else:
            components = nx.connected_components(graph)
        main = max(components, key=len, default=set())
        excluded = graph.number_of_nodes() - len(main)
        if excluded:
            logger.info(
                "Road graph (%s): %d of %d nodes lie outside the main component "
                "and are excluded from snapping/routing.",
                mode, excluded, graph.number_of_nodes(),
            )
        return set(main)

    def _routable_node(self, location: Location, mode: str) -> int:
        """Resolve a location to a node inside the main routable component.

        Instance files may carry node ids snapped to disconnected islands;
        those are re-snapped by coordinates instead of trusted blindly.
        """
        main = self._main_nodes.get(mode)
        node = location.node_id
        if node is not None and (not main or node in main):
            return node
        return self.nearest_node(location, mode)

    def _graph_path(self, mode: str) -> Path:
        map_cfg = self.config.get("map", {})
        configured_key = f"{mode}_graph_file"
        if map_cfg.get(configured_key):
            configured_path = Path(map_cfg[configured_key])
            if configured_path.is_absolute():
                return configured_path
            if configured_path.parent == Path("."):
                return self._map_dir / configured_path
            return self._repo_root / configured_path
        safe_name = self._place.replace(" ", "_").replace(",", "").replace("/", "_")
        return self._map_dir / f"{safe_name}_{mode}.pkl"

    def _resolve_map_dir(self, config: dict) -> Path:
        map_cfg = config.get("map", {})
        if map_cfg.get("graph_dir"):
            return self._resolve_repo_path(map_cfg["graph_dir"])
        return self._repo_root / "data" / "maps"

    def _load_graph_file(self, path: Path, mode: str):
        if not path.is_file():
            raise FileNotFoundError(
                f"Map graph file for mode={mode} does not exist: {path}. "
                "Put the prebuilt map pkl under data/maps or set map.graph_dir."
            )
        logger.info("Loading %s road graph from %s", mode, path)
        with path.open("rb") as f:
            return pickle.load(f)

    def _resolve_repo_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self._repo_root / path

    def _compute_bounds(self):
        if self.graph is None:
            return
        lats = [d["y"] for _, d in self.graph.nodes(data=True)]
        lons = [d["x"] for _, d in self.graph.nodes(data=True)]
        self._bounds = {
            "min_lat": min(lats), "max_lat": max(lats),
            "min_lon": min(lons), "max_lon": max(lons),
        }
