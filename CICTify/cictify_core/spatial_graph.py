import json
import math
import re
from collections import defaultdict
from datetime import datetime
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

from .config import SPATIAL_GRAPH_PATH


class SpatialGraphStore:
    def __init__(self) -> None:
        self._path = SPATIAL_GRAPH_PATH
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._nodes = {}
            self._edges = []
            return

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
            self._nodes = {str(node.get("id")): node for node in nodes if node.get("id")}
            self._edges = payload.get("edges", []) if isinstance(payload, dict) else []
        except Exception:
            self._nodes = {}
            self._edges = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now().isoformat(),
            "nodes": list(self._nodes.values()),
            "edges": self._edges,
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        val = str(value or "").strip().lower()
        return val in {"true", "yes", "y", "1", "open", "available"}

    def _merge_node(self, node: Dict, source_file: str) -> str:
        room_name = (node.get("name") or node.get("room") or "").strip() or "Unnamed Room"
        node_id = str(node.get("id") or self._slug(room_name) or f"room-{len(self._nodes) + 1}")

        existing = self._nodes.get(node_id, {})
        aliases = set(existing.get("aliases", []))
        for alias in node.get("aliases", []):
            alias_str = str(alias).strip()
            if alias_str:
                aliases.add(alias_str)
        if room_name:
            aliases.add(room_name)

        merged = {
            "id": node_id,
            "name": room_name,
            "building": (node.get("building") or existing.get("building") or "Unknown Building").strip(),
            "floor": (node.get("floor") or existing.get("floor") or "Unknown Floor").strip(),
            "kind": (node.get("kind") or existing.get("kind") or "room").strip().lower(),
            "has_open_pc": self._as_bool(node.get("has_open_pc", existing.get("has_open_pc", False))),
            "x": node.get("x", existing.get("x")),
            "y": node.get("y", existing.get("y")),
            "aliases": sorted(aliases),
            "source_file": source_file or existing.get("source_file", ""),
            "updated_at": datetime.now().isoformat(),
        }
        self._nodes[node_id] = merged
        return node_id

    def add_graph(self, graph: Dict, *, source_file: str) -> Dict:
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        edges = graph.get("edges", []) if isinstance(graph, dict) else []

        added_nodes = 0
        node_ids = set(self._nodes.keys())
        for node in nodes:
            node_id = self._merge_node(node if isinstance(node, dict) else {}, source_file)
            if node_id not in node_ids:
                added_nodes += 1
                node_ids.add(node_id)

        existing_edge_keys = {
            (str(e.get("from", "")), str(e.get("to", "")), str(e.get("door", "")))
            for e in self._edges
        }
        added_edges = 0
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            frm = str(edge.get("from") or "").strip()
            to = str(edge.get("to") or "").strip()
            if not frm or not to:
                continue
            if frm not in self._nodes or to not in self._nodes:
                continue
            key = (frm, to, str(edge.get("door") or ""))
            if key in existing_edge_keys:
                continue
            dist = edge.get("weight")
            if dist is None:
                dist = edge.get("distance")
            try:
                weight = float(dist)
            except Exception:
                weight = 1.0
            self._edges.append(
                {
                    "from": frm,
                    "to": to,
                    "door": str(edge.get("door") or "").strip(),
                    "weight": max(0.1, weight),
                    "source_file": source_file,
                }
            )
            existing_edge_keys.add(key)
            added_edges += 1

        self._save()
        return {
            "added_nodes": added_nodes,
            "added_edges": added_edges,
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
        }

    @staticmethod
    def _terms(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]{2,}", (text or "").lower())

    def _match_node_id(self, hint: str) -> Optional[str]:
        hint = (hint or "").strip()
        if not hint:
            return None
        terms = self._terms(hint)
        if not terms:
            return None

        best: Tuple[int, Optional[str]] = (-1, None)
        for node_id, node in self._nodes.items():
            hay = " ".join([node.get("name", "")] + list(node.get("aliases", []))).lower()
            score = sum(2 if term == hay else 1 for term in terms if term in hay)
            if score > best[0]:
                best = (score, node_id)

        return best[1] if best[0] > 0 else None

    def upsert_room_status(self, *, room_name: str, building: str, floor: str, has_open_pc: bool, source_file: str) -> Dict:
        existing_id = self._match_node_id(room_name)
        node = {
            "id": existing_id or "",
            "name": room_name,
            "building": building,
            "floor": floor,
            "kind": "lab" if "lab" in (room_name or "").lower() else "room",
            "has_open_pc": has_open_pc,
            "aliases": [room_name],
        }
        node_id = self._merge_node(node, source_file)
        self._save()
        return {
            "node_id": node_id,
            "room_name": self._nodes.get(node_id, {}).get("name", room_name),
            "has_open_pc": self._nodes.get(node_id, {}).get("has_open_pc", has_open_pc),
        }

    def _adjacency(self) -> Dict[str, List[Tuple[str, float]]]:
        graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for edge in self._edges:
            frm = str(edge.get("from") or "")
            to = str(edge.get("to") or "")
            if not frm or not to:
                continue
            try:
                w = float(edge.get("weight", 1.0))
            except Exception:
                w = 1.0
            graph[frm].append((to, w))
            graph[to].append((frm, w))
        return graph

    def shortest_path(self, start_id: str, goal_id: str) -> Optional[Dict]:
        if start_id not in self._nodes or goal_id not in self._nodes:
            return None
        if start_id == goal_id:
            node = self._nodes[start_id]
            return {
                "distance": 0.0,
                "path_ids": [start_id],
                "path_names": [node.get("name", start_id)],
            }

        adjacency = self._adjacency()
        dist = {start_id: 0.0}
        prev: Dict[str, Optional[str]] = {start_id: None}
        heap: List[Tuple[float, str]] = [(0.0, start_id)]

        while heap:
            cur_dist, cur = heappop(heap)
            if cur == goal_id:
                break
            if cur_dist > dist.get(cur, math.inf):
                continue

            for nxt, weight in adjacency.get(cur, []):
                candidate = cur_dist + weight
                if candidate < dist.get(nxt, math.inf):
                    dist[nxt] = candidate
                    prev[nxt] = cur
                    heappush(heap, (candidate, nxt))

        if goal_id not in dist:
            return None

        path_ids = []
        cursor = goal_id
        while cursor is not None:
            path_ids.append(cursor)
            cursor = prev.get(cursor)
        path_ids.reverse()

        return {
            "distance": round(dist[goal_id], 2),
            "path_ids": path_ids,
            "path_names": [self._nodes[n].get("name", n) for n in path_ids],
        }

    def answer_navigation_query(self, question: str) -> Optional[str]:
        q = (question or "").lower()
        if not self._nodes:
            return None

        nearest_terms = ["nearest", "closest", "pinakamalapit", "malapit"]
        pc_terms = ["pc", "computer", "available pc", "open pc", "bakanteng pc", "vacant pc"]
        route_terms = ["path", "route", "how do i get", "how to get", "papunta", "paano pumunta"]

        nearest_lab_query = (any(t in q for t in nearest_terms) and "lab" in q) and any(t in q for t in pc_terms)
        route_query = any(t in q for t in route_terms)

        if not nearest_lab_query and not route_query:
            return None

        start_hint = ""
        from_match = re.search(r"from\s+([a-zA-Z0-9\-\s]+)", q)
        if from_match:
            start_hint = from_match.group(1).strip()

        start_id = self._match_node_id(start_hint)
        if not start_id:
            start_id = self._match_node_id("entrance")
        if not start_id and self._nodes:
            start_id = next(iter(self._nodes.keys()))

        if not start_id:
            return None

        if nearest_lab_query:
            candidates = []
            for node_id, node in self._nodes.items():
                kind = str(node.get("kind") or "").lower()
                is_lab = "lab" in kind or "laboratory" in kind
                if not is_lab:
                    continue
                if not self._as_bool(node.get("has_open_pc", False)):
                    continue
                path = self.shortest_path(start_id, node_id)
                if path:
                    candidates.append((path["distance"], node_id, path))

            if not candidates:
                return (
                    "I found the spatial graph, but no lab is currently tagged with open PCs. "
                    "You can ingest room status images/descriptions to enrich this."
                )

            candidates.sort(key=lambda item: item[0])
            distance, target_id, route = candidates[0]
            target = self._nodes.get(target_id, {})
            return (
                f"Nearest lab with an open PC: {target.get('name', target_id)} "
                f"({target.get('building', 'Unknown Building')}, {target.get('floor', 'Unknown Floor')}).\n"
                f"Estimated path cost: {distance}\n"
                f"Route: {' -> '.join(route['path_names'])}"
            )

        # Generic route request: try to infer destination after 'to'.
        to_match = re.search(r"to\s+([a-zA-Z0-9\-\s]+)", q)
        if not to_match:
            return None

        target_hint = to_match.group(1).strip()
        target_id = self._match_node_id(target_hint)
        if not target_id:
            return None

        route = self.shortest_path(start_id, target_id)
        if not route:
            return "I could not find a connected path between those rooms in the current floorplan graph."

        target = self._nodes.get(target_id, {})
        return (
            f"Best route to {target.get('name', target_id)}: {' -> '.join(route['path_names'])}\n"
            f"Estimated path cost: {route['distance']}"
        )

    def stats(self) -> Dict:
        return {"nodes": len(self._nodes), "edges": len(self._edges)}
