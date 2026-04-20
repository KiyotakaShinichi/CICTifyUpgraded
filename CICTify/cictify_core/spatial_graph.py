import json
import math
import re
from collections import defaultdict
from datetime import datetime
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

from .config import CURATED_CORPUS_PATH, FLOORPLAN_CONTEXT_PATH, SPATIAL_GRAPH_PATH


class SpatialGraphStore:
    def __init__(self) -> None:
        self._path = SPATIAL_GRAPH_PATH
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Dict] = []
        self._load()
        if not self._nodes:
            boot = self._bootstrap_from_curated_corpus()
            if boot.get("added_nodes", 0) or boot.get("added_edges", 0):
                self._save()

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
    def _room_aliases(name: str) -> List[str]:
        aliases = set()
        clean = re.sub(r"\s+", " ", (name or "").strip())
        if not clean:
            return []
        aliases.add(clean)
        compact = clean.replace(" ", "")
        aliases.add(compact)

        for letters, digits in re.findall(r"\b([A-Za-z]{1,8})\s*(\d{1,3})\b", clean):
            aliases.add(f"{letters}{digits}")
            aliases.add(f"{letters} {digits}")

        lowered = clean.lower()
        if lowered.startswith("cict "):
            aliases.add(clean[5:].strip())
        if "office" in lowered:
            aliases.add(lowered.replace("office", "").strip().title())

        return sorted(a for a in aliases if a)

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
        for alias in self._room_aliases(room_name):
            aliases.add(alias)

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

    @staticmethod
    def _looks_room_like(name: str) -> bool:
        text = (name or "").strip().lower()
        if not text:
            return False
        if len(text) > 70:
            return False
        if text in {"references", "pimentel hall", "rooms", "room", "hallway", "building"}:
            return False
        room_keywords = {
            "room", "lab", "office", "hall", "avr", "acad", "dean", "faculty", "ojt",
            "networking", "server", "ideation", "conference", "center", "ct", "sdl", "it", "a",
        }
        if any(word in text for word in room_keywords):
            return True
        return bool(re.search(r"\d", text))

    @staticmethod
    def _normalize_room_name(name: str) -> str:
        text = (name or "").replace("\u2019", "'")
        text = re.sub(r"^[\u25cf\u2022\-*\s]+", "", text)
        text = re.sub(r"\s+", " ", text).strip(" .,:;-")
        return text

    @staticmethod
    def _floor_label(text: str) -> str:
        match = re.search(r"\b(\d)(?:st|nd|rd|th)\s+floor\b", (text or "").lower())
        if match:
            return f"{match.group(1)}F"
        return "Unknown Floor"

    @staticmethod
    def _floor_index(label: str) -> Optional[int]:
        match = re.search(r"(\d+)", str(label or ""))
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    def _add_edge(self, frm: str, to: str, weight: float, source_file: str, door: str = "") -> None:
        if not frm or not to or frm == to:
            return
        if frm not in self._nodes or to not in self._nodes:
            return
        key_a = (frm, to, door)
        key_b = (to, frm, door)
        for edge in self._edges:
            existing = (str(edge.get("from", "")), str(edge.get("to", "")), str(edge.get("door", "")))
            if existing == key_a or existing == key_b:
                return
        self._edges.append(
            {
                "from": frm,
                "to": to,
                "door": door,
                "weight": max(0.1, float(weight)),
                "source_file": source_file,
            }
        )

    def _extract_relation_targets(self, description: str) -> List[Tuple[str, str, float]]:
        text = (description or "").replace("\u2019", "'")
        relations: List[Tuple[str, str, float]] = []

        between = re.search(
            r"\bbetween\s+([A-Za-z0-9/\- ]{1,50}?)\s+and\s+([A-Za-z0-9/\- ]{1,50})(?:\b|[.,;])",
            text,
            flags=re.IGNORECASE,
        )
        if between:
            relations.append(("between", self._normalize_room_name(between.group(1)), 1.2))
            relations.append(("between", self._normalize_room_name(between.group(2)), 1.2))

        for pat in [r"\bbeside\s+([A-Za-z0-9/\- ]{1,50})(?:\b|[.,;])", r"\bnext\s+to\s+([A-Za-z0-9/\- ]{1,50})(?:\b|[.,;])"]:
            for match in re.finditer(pat, text, flags=re.IGNORECASE):
                relations.append(("adjacent", self._normalize_room_name(match.group(1)), 1.0))

        for pat in [r"\binside\s+the\s+([A-Za-z0-9/\- ]{1,50})(?:\b|[.,;])", r"\binside\s+([A-Za-z0-9/\- ]{1,50})(?:\b|[.,;])"]:
            for match in re.finditer(pat, text, flags=re.IGNORECASE):
                relations.append(("inside", self._normalize_room_name(match.group(1)), 0.8))

        return [(kind, target, w) for kind, target, w in relations if self._looks_room_like(target)]

    def _bootstrap_from_curated_corpus(self) -> Dict:
        if not CURATED_CORPUS_PATH.exists():
            return {"added_nodes": 0, "added_edges": 0, "total_nodes": len(self._nodes), "total_edges": len(self._edges)}

        try:
            payload = json.loads(CURATED_CORPUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"added_nodes": 0, "added_edges": 0, "total_nodes": len(self._nodes), "total_edges": len(self._edges)}

        chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        docs = payload.get("documents", []) if isinstance(payload, dict) else []

        texts: List[Tuple[str, str]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            source = str(doc.get("source_file") or "")
            title = str(doc.get("title") or "").lower()
            if "cict rooms" in source.lower() or "cict rooms" in title or "floor plan" in title:
                body = str(doc.get("normalized_text") or "")
                if body:
                    texts.append((source, body))

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            source = str(chunk.get("source_file") or "")
            content = str(chunk.get("content") or "")
            if not content:
                continue
            low = content.lower()
            source_low = source.lower()
            if (
                "cict rooms" in source_low
                or "cict-rooms" in source_low
                or ("student handbook" in source_low and "evacuation" in low)
            ):
                texts.append((source, content))

        before_nodes = len(self._nodes)
        before_edges = len(self._edges)
        current_building = "Pimentel Hall"
        current_floor = "Unknown Floor"
        stair_nodes: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for source, text in texts:
            normalized = text.replace("\u25cf", "\n\u25cf ").replace("\u2022", "\n\u2022 ")
            lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
            for line in lines:
                lower = line.lower()
                if "pimentel hall" in lower:
                    current_building = "Pimentel Hall"
                if "nstp building" in lower:
                    current_building = "NSTP Building"

                floor_label = self._floor_label(line)
                if floor_label != "Unknown Floor":
                    current_floor = floor_label

                room_match = re.match(r"^[\u25cf\u2022\-*\s]*([A-Za-z][A-Za-z0-9/ .'-]{1,60}?)\s*[\u2013\-]\s*(.+)$", line)
                if not room_match:
                    continue

                room_name = self._normalize_room_name(room_match.group(1))
                desc = self._normalize_room_name(room_match.group(2))
                if not self._looks_room_like(room_name):
                    continue

                explicit_floor = self._floor_label(desc)
                floor = explicit_floor if explicit_floor != "Unknown Floor" else current_floor
                building = current_building
                if "nstp building" in desc.lower():
                    building = "NSTP Building"
                elif "pimentel hall" in desc.lower():
                    building = "Pimentel Hall"

                room_id = self._merge_node(
                    {
                        "name": room_name,
                        "building": building,
                        "floor": floor,
                        "kind": "lab" if "lab" in room_name.lower() else "room",
                        "aliases": self._room_aliases(room_name),
                    },
                    source,
                )

                for _, target_name, weight in self._extract_relation_targets(desc):
                    target_id = self._merge_node(
                        {
                            "name": target_name,
                            "building": building,
                            "floor": floor,
                            "kind": "lab" if "lab" in target_name.lower() else "room",
                            "aliases": self._room_aliases(target_name),
                        },
                        source,
                    )
                    self._add_edge(room_id, target_id, weight, source)

                stair_kind = ""
                if "central stairs" in lower or "central stairs" in desc.lower():
                    stair_kind = "central"
                elif "rear stairs" in lower or "rear stairs" in desc.lower():
                    stair_kind = "rear"
                elif "staircase" in lower or "stairs" in lower or "stairs" in desc.lower():
                    stair_kind = "main"

                if stair_kind:
                    stairs_name = f"{building} {floor} {stair_kind.title()} Stairs"
                    stairs_id = self._merge_node(
                        {
                            "name": stairs_name,
                            "building": building,
                            "floor": floor,
                            "kind": "stairs",
                            "aliases": [f"{stair_kind} stairs", "stairs", f"{building} stairs"],
                        },
                        source,
                    )
                    self._add_edge(room_id, stairs_id, 1.1, source)
                    stair_nodes[(building, stair_kind)].append(stairs_id)

        for (building, stair_kind), node_ids in stair_nodes.items():
            unique_ids = sorted(set(node_ids), key=lambda node_id: self._floor_index(self._nodes.get(node_id, {}).get("floor", "")) or 99)
            for idx in range(len(unique_ids) - 1):
                a = unique_ids[idx]
                b = unique_ids[idx + 1]
                self._add_edge(a, b, 2.0, f"bootstrap:{building}:{stair_kind}")

        return {
            "added_nodes": len(self._nodes) - before_nodes,
            "added_edges": len(self._edges) - before_edges,
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
        }

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

    @staticmethod
    def _norm_for_match(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (text or "").lower())

    def _match_node_id(self, hint: str) -> Optional[str]:
        hint = (hint or "").strip()
        if not hint:
            return None
        hint_clean = re.sub(r"\s+", " ", hint).strip()
        hint_lower = hint_clean.lower()
        hint_norm = self._norm_for_match(hint_clean)

        # 1) Exact match against aliases/names (case-insensitive).
        for node_id, node in self._nodes.items():
            candidates = [node.get("name", "")] + list(node.get("aliases", []))
            for alias in candidates:
                alias_text = str(alias or "").strip()
                if not alias_text:
                    continue
                if alias_text.lower() == hint_lower:
                    return node_id

        # 2) Exact normalized match (ignores spaces/punctuation).
        for node_id, node in self._nodes.items():
            candidates = [node.get("name", "")] + list(node.get("aliases", []))
            for alias in candidates:
                alias_text = str(alias or "").strip()
                if not alias_text:
                    continue
                if self._norm_for_match(alias_text) == hint_norm:
                    return node_id

        terms = self._terms(hint)
        if not terms:
            return None

        best: Tuple[int, Optional[str]] = (-1, None)
        for node_id, node in self._nodes.items():
            alias_pool = [str(node.get("name", "") or "")] + [str(a or "") for a in node.get("aliases", [])]
            hay = " ".join(alias_pool).lower()
            hay_terms = set(self._terms(hay))

            score = 0
            for term in terms:
                # Whole-token match is stronger than substring match.
                if term in hay_terms:
                    score += 3
                elif term in hay:
                    score += 1

            # Bonus when normalized query is contained as contiguous sequence.
            if hint_norm and any(hint_norm in self._norm_for_match(alias) for alias in alias_pool):
                score += 3

            if score > best[0]:
                best = (score, node_id)

        return best[1] if best[0] > 0 else None

    def _building_aliases(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        for node in self._nodes.values():
            building = str(node.get("building") or "").strip()
            if not building:
                continue
            norm = self._norm_for_match(building)
            if norm:
                aliases[norm] = building

            parts = [p for p in re.split(r"\s+", building.lower()) if p and p not in {"building", "hall"}]
            for part in parts:
                part_norm = self._norm_for_match(part)
                if len(part_norm) >= 4:
                    aliases[part_norm] = building

        # Handbook/campus map derived building aliases (pages 101-110 in paper handbook).
        handbook_buildings = {
            "Pimentel Hall": ["pimentel hall", "pimentel"],
            "NSTP Building": ["nstp building", "nstp"],
            "Alvarado Hall": ["alvarado hall", "alvarado"],
            "Valencia Hall": ["valencia hall", "valencia"],
            "Federizo Hall": ["federizo hall", "federizo"],
            "Flores Hall": ["flores hall", "administration hall", "admin building", "administration building"],
            "Carpio Hall": ["carpio hall", "carpio"],
            "Natividad Hall": ["natividad hall", "natividad"],
            "Alumni Building": ["alumni building"],
            "Academic Building": ["academic building"],
            "Round Hall": ["round hall"],
            "Laboratory Building": ["laboratory building", "lab building"],
            "Activity Center": ["activity center"],
            "Hostel": ["hostel", "dorm", "dormitory"],
        }
        for canonical, alias_list in handbook_buildings.items():
            aliases[self._norm_for_match(canonical)] = canonical
            for alias in alias_list:
                aliases[self._norm_for_match(alias)] = canonical

        # Helpful manual aliases for common wording.
        if "pimentelhall" in aliases:
            aliases["pimentel"] = aliases["pimentelhall"]
        if "nstpbuilding" in aliases:
            aliases["nstp"] = aliases["nstpbuilding"]

        return aliases

    def _match_building_name(self, hint: str) -> Optional[str]:
        hint_norm = self._norm_for_match(hint)
        if not hint_norm:
            return None
        aliases = self._building_aliases()

        if hint_norm in aliases:
            return aliases[hint_norm]

        for alias_norm, building in aliases.items():
            if alias_norm and (alias_norm in hint_norm or hint_norm in alias_norm):
                return building
        return None

    def _nodes_for_building(self, building: str) -> List[str]:
        target = (building or "").strip().lower()
        if not target:
            return []
        return [
            node_id
            for node_id, node in self._nodes.items()
            if str(node.get("building") or "").strip().lower() == target
        ]

    def _best_route_between_buildings(self, building_a: str, building_b: str) -> Optional[Dict]:
        ids_a = self._nodes_for_building(building_a)
        ids_b = self._nodes_for_building(building_b)
        if not ids_a or not ids_b:
            return None

        best: Optional[Dict] = None
        for a in ids_a:
            for b in ids_b:
                route = self.shortest_path(a, b)
                if not route:
                    continue
                if best is None or route["distance"] < best["distance"]:
                    best = {
                        "distance": route["distance"],
                        "from_id": a,
                        "to_id": b,
                        "route": route,
                    }
        return best

    def _landmarks_for_building(self, building: str, max_items: int = 4) -> List[str]:
        if not FLOORPLAN_CONTEXT_PATH.exists() or not building:
            return []
        try:
            payload = json.loads(FLOORPLAN_CONTEXT_PATH.read_text(encoding="utf-8"))
            records = payload.get("records", []) if isinstance(payload, dict) else []
        except Exception:
            return []

        target_norm = self._norm_for_match(building)
        if not target_norm:
            return []

        known_tokens = {
            "park",
            "gate",
            "court",
            "library",
            "canteen",
            "gym",
            "gymnasium",
            "coop",
            "parking",
            "tank",
            "office",
            "center",
        }

        found: List[str] = []
        seen = set()
        for rec in records:
            if not isinstance(rec, dict):
                continue
            text = str(rec.get("ocr_text") or "")
            if not text:
                continue
            text_norm = self._norm_for_match(text)
            if target_norm not in text_norm:
                continue

            for line in text.splitlines():
                item = re.sub(r"^[\s*\-\u2022\d\.\)]+", "", line).strip(" :.-")
                if not item:
                    continue
                item_low = item.lower()
                if item_low.startswith("building/hall") or item_low.startswith("wayfinding"):
                    continue
                if self._norm_for_match(item) == target_norm:
                    continue

                if any(tok in item_low for tok in known_tokens) or "hall" in item_low or "building" in item_low:
                    clean = re.sub(r"\s+", " ", item).strip()
                    norm = self._norm_for_match(clean)
                    if norm and norm != target_norm and norm not in seen:
                        seen.add(norm)
                        found.append(clean)
                        if len(found) >= max_items:
                            return found

        return found

    @staticmethod
    def _looks_building_hint(text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return False
        return any(token in low for token in ["building", "hall", "hostel", "dorm", "campus", "pimentel", "nstp"])

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
        route_terms = ["path", "route", "how do i get", "how to get", "papunta", "paano pumunta", "directions"]
        near_terms = ["near", "nearby", "beside", "next to", "katabi", "malapit"]
        where_terms = ["where is", "nasaan", "asan", "nasan"]

        nearest_lab_query = (any(t in q for t in nearest_terms) and "lab" in q) and any(t in q for t in pc_terms)
        route_query = any(t in q for t in route_terms)
        near_query = any(t in q for t in near_terms)
        where_query = any(t in q for t in where_terms)

        if not nearest_lab_query and not route_query and not near_query and not where_query:
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

        near_match = re.search(
            r"(?:is|are)?\s*([a-z0-9/\-\s']+?)\s+(?:near|nearby|beside|next\s+to|katabi(?:\s+ng)?|malapit(?:\s+sa)?)\s+([a-z0-9/\-\s']+)",
            q,
            flags=re.IGNORECASE,
        )
        if near_match:
            left_hint = near_match.group(1).strip(" ?!.,")
            right_hint = near_match.group(2).strip(" ?!.,")
            if not re.fullmatch(r"(?:what|which|where)(?:\s+is)?", left_hint.lower()):
                left_building = self._match_building_name(left_hint)
                right_building = self._match_building_name(right_hint)
                building_mode = self._looks_building_hint(left_hint) or self._looks_building_hint(right_hint)

                if building_mode:
                    if left_building and right_building:
                        if left_building.lower() == right_building.lower():
                            return f"Yes. {left_building} and {right_building} refer to the same building."

                        building_route = self._best_route_between_buildings(left_building, right_building)
                        if building_route:
                            route = building_route["route"]
                            is_near = route["distance"] <= 4.0
                            if is_near:
                                return (
                                    f"Yes, they are relatively near in the mapped campus graph. "
                                    f"Closest mapped path between {left_building} and {right_building}: {' -> '.join(route['path_names'])}\n"
                                    f"Estimated path cost: {route['distance']}"
                                )
                            return (
                                f"They are connected in the mapped campus graph but not immediately near. "
                                f"Closest mapped path between {left_building} and {right_building}: {' -> '.join(route['path_names'])}\n"
                                f"Estimated path cost: {route['distance']}"
                            )

                        return (
                            f"I know both buildings ({left_building} and {right_building}) but there is no inter-building connector in the current graph yet. "
                            "Upload or ingest campus pathways/floorplans to enable step-by-step cross-building directions."
                        )

                    known_buildings = sorted({str(n.get("building") or "").strip() for n in self._nodes.values() if str(n.get("building") or "").strip()})
                    unknown = right_hint if left_building and not right_building else left_hint
                    if left_building or right_building:
                        return (
                            f"I could not find '{unknown}' as a mapped building yet. "
                            f"Known buildings: {', '.join(known_buildings)}."
                        )

                left_id = self._match_node_id(left_hint)
                right_id = self._match_node_id(right_hint)
                if left_id and right_id:
                    path = self.shortest_path(left_id, right_id)
                    left_name = self._nodes[left_id].get("name", left_id)
                    right_name = self._nodes[right_id].get("name", right_id)
                    if path:
                        is_near = path["distance"] <= 2.2 or len(path["path_ids"]) <= 3
                        if is_near:
                            return (
                                f"Yes. {left_name} is near {right_name}.\n"
                                f"Estimated path cost: {path['distance']}\n"
                                f"Route: {' -> '.join(path['path_names'])}"
                            )
                        return (
                            f"Not immediately near. {left_name} and {right_name} are connected but farther apart.\n"
                            f"Estimated path cost: {path['distance']}\n"
                            f"Route: {' -> '.join(path['path_names'])}"
                        )
                    left_node = self._nodes.get(left_id, {})
                    right_node = self._nodes.get(right_id, {})
                    left_loc = f"{left_node.get('building', 'Unknown Building')}, {left_node.get('floor', 'Unknown Floor')}"
                    right_loc = f"{right_node.get('building', 'Unknown Building')}, {right_node.get('floor', 'Unknown Floor')}"
                    return (
                        f"I cannot confirm direct nearness yet because the graph has no connected path between {left_name} and {right_name}. "
                        f"Known locations: {left_name} ({left_loc}); {right_name} ({right_loc})."
                    )

        single_near_match = re.search(r"(?:what|which|where)\s+(?:is|are)?\s*(?:near|nearby\s+to|close\s+to)\s+([a-z0-9/\-\s']+)", q, flags=re.IGNORECASE)
        if single_near_match:
            target_hint = single_near_match.group(1).strip(" ?!.,")
            target_building = self._match_building_name(target_hint)
            if target_building:
                landmarks = self._landmarks_for_building(target_building)
                if landmarks:
                    return f"Nearby landmarks around {target_building} (from campus map references): {', '.join(landmarks)}."
                return f"I recognize {target_building}, but I do not have enough nearby landmark links yet in the current graph."

            target_id = self._match_node_id(target_hint)
            if target_id:
                neighbors = [
                    self._nodes.get(nxt, {}).get("name", nxt)
                    for nxt, _ in sorted(self._adjacency().get(target_id, []), key=lambda item: item[1])[:4]
                ]
                target_name = self._nodes.get(target_id, {}).get("name", target_id)
                if neighbors:
                    return f"Nearby places around {target_name}: {', '.join(neighbors)}."
                return f"I recognize {target_name}, but no nearby nodes are linked yet in the current graph."

        if where_query:
            where_match = re.search(r"(?:where\s+is|nasaan\s+ang|nasaan|asan|nasan)\s+(.+)$", q, flags=re.IGNORECASE)
            target_hint = where_match.group(1).strip(" ?!.,") if where_match else ""
            target_building = self._match_building_name(target_hint)
            if target_building and self._looks_building_hint(target_hint):
                floors = sorted({str(n.get("floor") or "Unknown Floor") for n in self._nodes.values() if str(n.get("building") or "").strip().lower() == target_building.lower()})
                landmarks = self._landmarks_for_building(target_building)
                floor_text = ", ".join(floors) if floors else "Unknown Floor"
                if landmarks:
                    return (
                        f"{target_building} appears in the spatial graph with mapped areas on: {floor_text}. "
                        f"Nearby landmarks (campus map): {', '.join(landmarks)}."
                    )
                return f"{target_building} appears in the spatial graph with mapped areas on: {floor_text}."

            target_id = self._match_node_id(target_hint)
            if target_id:
                node = self._nodes.get(target_id, {})
                neighbors = [self._nodes.get(nxt, {}).get("name", nxt) for nxt, _ in sorted(self._adjacency().get(target_id, []), key=lambda item: item[1])[:3]]
                neighbor_text = f" Nearby: {', '.join(neighbors)}." if neighbors else ""
                return (
                    f"{node.get('name', target_id)} is in {node.get('building', 'Unknown Building')}, {node.get('floor', 'Unknown Floor')}."
                    f"{neighbor_text}"
                )
            if target_building:
                floors = sorted({str(n.get("floor") or "Unknown Floor") for n in self._nodes.values() if str(n.get("building") or "").strip().lower() == target_building.lower()})
                floor_text = ", ".join(floors) if floors else "Unknown Floor"
                return f"{target_building} appears in the spatial graph with mapped areas on: {floor_text}."

        # Generic route request: try to infer destination after 'to'.
        target_hint = ""
        from_to_match = re.search(r"from\s+([a-zA-Z0-9/\-\s]+?)\s+to\s+([a-zA-Z0-9/\-\s]+)", q, flags=re.IGNORECASE)
        if from_to_match:
            start_hint = from_to_match.group(1).strip()
            target_hint = from_to_match.group(2).strip()
            start_id = self._match_node_id(start_hint) or start_id
        else:
            to_match = re.search(r"to\s+([a-zA-Z0-9/\-\s]+)", q, flags=re.IGNORECASE)
            if to_match:
                target_hint = to_match.group(1).strip()

        if not target_hint:
            return None

        target_id = self._match_node_id(target_hint)

        start_building = self._match_building_name(start_hint) if start_hint else None
        target_building = self._match_building_name(target_hint) if target_hint else None
        building_route_mode = self._looks_building_hint(start_hint) or self._looks_building_hint(target_hint)

        if building_route_mode and start_building and target_building:
            building_route = self._best_route_between_buildings(start_building, target_building)
            if not building_route:
                return (
                    f"Route unavailable between {start_building} and {target_building} in the current graph. "
                    "Upload or ingest campus pathways/floorplans to enable cross-building directions."
                )
            route = building_route["route"]
            return (
                f"Best mapped route from {start_building} to {target_building}: {' -> '.join(route['path_names'])}\n"
                f"Estimated path cost: {route['distance']}"
            )

        if building_route_mode and (start_building or target_building) and not (start_building and target_building):
            known_buildings = sorted({str(n.get("building") or "").strip() for n in self._nodes.values() if str(n.get("building") or "").strip()})
            missing_hint = target_hint if start_building and not target_building else start_hint
            return f"I could not find '{missing_hint}' as a mapped building yet. Known buildings: {', '.join(known_buildings)}."

        if not target_id:
            return None

        route = self.shortest_path(start_id, target_id)
        if not route:
            start_node = self._nodes.get(start_id, {})
            target_node = self._nodes.get(target_id, {})
            start_name = start_node.get("name", start_id)
            target_name = target_node.get("name", target_id)
            start_loc = f"{start_node.get('building', 'Unknown Building')}, {start_node.get('floor', 'Unknown Floor')}"
            target_loc = f"{target_node.get('building', 'Unknown Building')}, {target_node.get('floor', 'Unknown Floor')}"
            return (
                f"Route unavailable between {start_name} and {target_name} in the current floorplan graph. "
                f"Known locations: {start_name} ({start_loc}); {target_name} ({target_loc})."
            )

        target = self._nodes.get(target_id, {})
        return (
            f"Best route to {target.get('name', target_id)}: {' -> '.join(route['path_names'])}\n"
            f"Estimated path cost: {route['distance']}"
        )

    def stats(self) -> Dict:
        return {"nodes": len(self._nodes), "edges": len(self._edges)}
