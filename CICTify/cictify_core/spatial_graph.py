import json
import math
import re
from collections import defaultdict
from datetime import datetime
from heapq import heappop, heappush
from typing import Dict, List, Optional, Tuple

from .config import CURATED_CORPUS_PATH, FLOORPLAN_CONTEXT_PATH, SPATIAL_GRAPH_PATH


CICT_PRIMARY_BUILDING_FLOORS = {
    "pimentel hall": {"3f", "4f"},
    "nstp building": {"1f"},
}

CAMPUS_PLACE_TOKENS = {
    "gate",
    "park",
    "building",
    "hall",
    "registrar",
    "hostel",
    "dorm",
    "gym",
    "gymnasium",
    "canteen",
    "library",
    "parking",
    "court",
    "center",
}


class SpatialGraphStore:
    def __init__(self) -> None:
        self._path = SPATIAL_GRAPH_PATH
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Dict] = []
        self._load()
        touched = self._ensure_alias_hints()
        touched = self._ensure_connectivity_hints() or touched
        campus_boot = self._bootstrap_campus_places_from_floorplan_context()
        touched = bool(campus_boot.get("added_nodes") or campus_boot.get("added_edges")) or touched
        if not self._nodes:
            boot = self._bootstrap_from_curated_corpus()
            if boot.get("added_nodes", 0) or boot.get("added_edges", 0):
                self._save()
        elif touched:
            self._save()

    @staticmethod
    def _clean_bullet_item(line: str) -> str:
        text = re.sub(r"^[\s*\-\u2022\u25cf\d\.)]+", "", str(line or "")).strip()
        text = text.replace("**", "")
        if ":" in text:
            text = text.split(":", 1)[0].strip()
        text = re.sub(r"\s+", " ", text).strip(" .,:;-")
        return text

    @staticmethod
    def _looks_campus_place(name: str) -> bool:
        low = str(name or "").strip().lower()
        if not low:
            return False
        if len(low) > 80:
            return False
        if low in {"buildings/halls", "buildings", "wayfinding landmarks", "landmarks"}:
            return False
        if "evacuation" in low or "plan" in low:
            return False
        if any(tok in low for tok in CAMPUS_PLACE_TOKENS):
            return True
        return bool(re.search(r"\b(hall|building|gate|park|court|hostel|library|gym|registrar)\b", low))

    def _extract_campus_places_from_ocr(self, text: str) -> List[str]:
        items: List[str] = []
        seen = set()
        for raw in str(text or "").splitlines():
            item = self._clean_bullet_item(raw)
            if not self._looks_campus_place(item):
                continue
            norm = self._norm_for_match(item)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            items.append(item)
        return items

    def _campus_hub_id(self) -> str:
        return self._merge_node(
            {
                "id": "campus-main-hub",
                "name": "BulSU Main Campus Hub",
                "building": "BulSU Main Campus",
                "floor": "Campus",
                "kind": "campus_place",
                "aliases": ["main campus", "campus hub", "bulsu main campus"],
            },
            "inferred:campus-hub",
        )

    def _building_entry_node(self, building: str) -> Optional[str]:
        candidates = []
        for node_id, node in self._nodes.items():
            if str(node.get("building") or "").strip().lower() != str(building or "").strip().lower():
                continue
            kind = str(node.get("kind") or "").strip().lower()
            name = str(node.get("name") or "").strip().lower()
            floor_idx = self._floor_index(str(node.get("floor") or ""))
            floor_rank = floor_idx if floor_idx is not None else 99
            stair_rank = 0 if ("stairs" in kind or "stairs" in name) else 1
            candidates.append((floor_rank, stair_rank, node_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _bootstrap_campus_places_from_floorplan_context(self) -> Dict:
        if not FLOORPLAN_CONTEXT_PATH.exists():
            return {"added_nodes": 0, "added_edges": 0}

        try:
            payload = json.loads(FLOORPLAN_CONTEXT_PATH.read_text(encoding="utf-8"))
            records = payload.get("records", []) if isinstance(payload, dict) else []
        except Exception:
            return {"added_nodes": 0, "added_edges": 0}

        before_nodes = len(self._nodes)
        before_edges = len(self._edges)

        hub_id = self._campus_hub_id()
        campus_node_ids: List[str] = []

        for rec in records:
            if not isinstance(rec, dict):
                continue
            floor = str(rec.get("floor") or "").strip().lower()
            title = str(rec.get("title") or "").strip().lower()
            source = str(rec.get("source_file") or "")
            is_campus_map = ("campus map" in floor) or ("evacuation map" in title)
            if not is_campus_map:
                continue

            building = str(rec.get("building") or "BulSU Main Campus").strip() or "BulSU Main Campus"
            for place in self._extract_campus_places_from_ocr(str(rec.get("ocr_text") or "")):
                node_id = self._merge_node(
                    {
                        "name": place,
                        "building": building,
                        "floor": "Campus",
                        "kind": "campus_place",
                        "aliases": [place],
                    },
                    source,
                )
                campus_node_ids.append(node_id)
                self._add_edge(hub_id, node_id, 2.2, source)

        # Ensure important non-floor campus POIs exist even if OCR misses them.
        for poi in ["Registrar", "Gate 1", "Gate 2", "Main Gate"]:
            node_id = self._merge_node(
                {
                    "name": poi,
                    "building": "BulSU Main Campus",
                    "floor": "Campus",
                    "kind": "campus_place",
                    "aliases": [poi],
                },
                "inferred:campus-poi-seed",
            )
            campus_node_ids.append(node_id)
            self._add_edge(hub_id, node_id, 1.8, "inferred:campus-poi-seed")

        unique_campus_nodes = sorted(set(campus_node_ids))
        for node_id in unique_campus_nodes:
            node = self._nodes.get(node_id, {})
            name = str(node.get("name") or "")
            target_building = self._match_building_name(name)
            if not target_building:
                continue
            entry_id = self._building_entry_node(target_building)
            if entry_id:
                self._add_edge(node_id, entry_id, 2.0, "inferred:campus-building-connector")

        # Keep campus traversable: link hub to every known building entry.
        connected_buildings = sorted({
            str(node.get("building") or "").strip()
            for node in self._nodes.values()
            if str(node.get("building") or "").strip()
        })
        for building in connected_buildings:
            if building.lower() == "bulsu main campus":
                continue
            entry_id = self._building_entry_node(building)
            if entry_id:
                self._add_edge(hub_id, entry_id, 3.0, "inferred:campus-hub-building")

        return {
            "added_nodes": len(self._nodes) - before_nodes,
            "added_edges": len(self._edges) - before_edges,
        }

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

        if "dean" in lowered:
            aliases.update({"Dean Office", "Dean's Office", "Deans Office", "Office of the Dean"})

        prog_lab_match = re.search(r"\bprog\s*lab\s*(\d+)\b", lowered)
        if prog_lab_match:
            num = prog_lab_match.group(1)
            aliases.update({
                f"Programming Laboratory {num}",
                f"Programming Lab {num}",
                f"Prog Lab {num}",
                f"ProgLab{num}",
            })

        return sorted(a for a in aliases if a)

    def _ensure_alias_hints(self) -> bool:
        touched = False
        for node in self._nodes.values():
            name = str(node.get("name") or "").strip()
            if not name:
                continue
            aliases = set(str(a).strip() for a in node.get("aliases", []) if str(a).strip())
            before = set(aliases)
            aliases.update(self._room_aliases(name))
            if aliases != before:
                node["aliases"] = sorted(aliases)
                touched = True
        return touched

    def _nodes_by_pattern(self, *, building: str, floor: str, pattern: str) -> List[str]:
        rgx = re.compile(pattern, flags=re.IGNORECASE)
        hits: List[str] = []
        for node_id, node in self._nodes.items():
            if str(node.get("building") or "").strip().lower() != building.lower():
                continue
            if str(node.get("floor") or "").strip().lower() != floor.lower():
                continue
            hay = f"{node.get('name', '')} {' '.join(str(a) for a in node.get('aliases', []))}"
            if rgx.search(hay):
                hits.append(node_id)
        return hits

    def _ensure_connectivity_hints(self) -> bool:
        before_edges = len(self._edges)

        floor_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for node_id, node in self._nodes.items():
            building = str(node.get("building") or "").strip()
            floor = str(node.get("floor") or "").strip()
            if not building or not floor:
                continue
            floor_groups[(building, floor)].append(node_id)

        for (building, floor), node_ids in floor_groups.items():
            stair_ids = [
                n
                for n in node_ids
                if "stairs" in str(self._nodes.get(n, {}).get("kind") or "").lower()
                or "stairs" in str(self._nodes.get(n, {}).get("name") or "").lower()
            ]
            for i in range(len(stair_ids)):
                for j in range(i + 1, len(stair_ids)):
                    self._add_edge(stair_ids[i], stair_ids[j], 1.8, f"inferred:{building}:{floor}:stairs")

            dean_nodes = self._nodes_by_pattern(building=building, floor=floor, pattern=r"\bdean")
            ojt_nodes = self._nodes_by_pattern(building=building, floor=floor, pattern=r"\bojt\b|department\s+heads")
            faculty_nodes = self._nodes_by_pattern(building=building, floor=floor, pattern=r"faculty\s+room")
            it1_nodes = self._nodes_by_pattern(building=building, floor=floor, pattern=r"\bit\s*1\b")

            for dean in dean_nodes:
                for ojt in ojt_nodes:
                    if dean != ojt:
                        self._add_edge(dean, ojt, 1.0, f"inferred:{building}:{floor}:dean-ojt")

            for i in range(len(ojt_nodes)):
                for j in range(i + 1, len(ojt_nodes)):
                    self._add_edge(ojt_nodes[i], ojt_nodes[j], 1.0, f"inferred:{building}:{floor}:ojt-merge")

            for ojt in ojt_nodes:
                for fac in faculty_nodes:
                    if ojt != fac:
                        self._add_edge(ojt, fac, 1.0, f"inferred:{building}:{floor}:ojt-faculty")

            for fac in faculty_nodes:
                for it1 in it1_nodes:
                    if fac != it1:
                        self._add_edge(fac, it1, 1.2, f"inferred:{building}:{floor}:faculty-it1")

        self._ensure_stair_chains()

        return len(self._edges) > before_edges

    def _ensure_stair_chains(self) -> None:
        stair_groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for node_id, node in self._nodes.items():
            kind = str(node.get("kind") or "").strip().lower()
            name = str(node.get("name") or "").strip()
            if "stairs" not in kind and "stairs" not in name.lower():
                continue
            building = str(node.get("building") or "").strip()
            floor = str(node.get("floor") or "").strip()
            if not building or not floor:
                continue
            stair_kind = "main"
            low = name.lower()
            if "central" in low:
                stair_kind = "central"
            elif "rear" in low:
                stair_kind = "rear"
            stair_groups[(building, stair_kind)].append(node_id)

        for (building, stair_kind), ids in stair_groups.items():
            floors_present = {}
            for node_id in ids:
                floor = str(self._nodes.get(node_id, {}).get("floor") or "")
                idx = self._floor_index(floor)
                if idx is None:
                    continue
                floors_present[idx] = node_id

            if len(floors_present) < 2:
                continue

            min_floor = min(floors_present.keys())
            max_floor = max(floors_present.keys())
            for level in range(min_floor, max_floor + 1):
                if level not in floors_present:
                    floor_label = f"{level}F"
                    stairs_name = f"{building} {floor_label} {stair_kind.title()} Stairs"
                    node_id = self._merge_node(
                        {
                            "name": stairs_name,
                            "building": building,
                            "floor": floor_label,
                            "kind": "stairs",
                            "aliases": [f"{stair_kind} stairs", f"{building} stairs", "stairs"],
                        },
                        f"inferred:{building}:{stair_kind}:virtual-floor",
                    )
                    floors_present[level] = node_id

            ordered = sorted(floors_present.items(), key=lambda kv: kv[0])
            for idx in range(len(ordered) - 1):
                floor_a, node_a = ordered[idx]
                floor_b, node_b = ordered[idx + 1]
                if floor_b - floor_a == 1:
                    self._add_edge(node_a, node_b, 1.8, f"inferred:{building}:{stair_kind}:chain")

    def _edge_allowed(self, frm: str, to: str) -> bool:
        a = self._nodes.get(frm, {})
        b = self._nodes.get(to, {})
        a_name = str(a.get("name") or "").lower()
        b_name = str(b.get("name") or "").lower()
        a_kind = str(a.get("kind") or "").lower()
        b_kind = str(b.get("kind") or "").lower()
        a_is_stairs = "stairs" in a_kind or "stairs" in a_name
        b_is_stairs = "stairs" in b_kind or "stairs" in b_name
        if a_is_stairs and b_is_stairs:
            a_build = str(a.get("building") or "").strip().lower()
            b_build = str(b.get("building") or "").strip().lower()
            if a_build == b_build:
                af = self._floor_index(str(a.get("floor") or ""))
                bf = self._floor_index(str(b.get("floor") or ""))
                if af is not None and bf is not None and abs(af - bf) > 1:
                    return False
        return True

    def _is_cict_indoor_node(self, node: Dict) -> bool:
        building = str(node.get("building") or "").strip().lower()
        floor = str(node.get("floor") or "").strip().lower()
        kind = str(node.get("kind") or "").strip().lower()
        name = str(node.get("name") or "").strip().lower()

        if "stairs" in kind or "stairs" in name:
            # Keep stairs only on CICT-authorized floors for each building.
            allowed = CICT_PRIMARY_BUILDING_FLOORS.get(building)
            if allowed is None:
                return False
            return floor in allowed

        if kind in {"room", "lab", "office"} or any(tok in name for tok in ["lab", "office", "room", "sdl", "acad", "it", "prog"]):
            allowed = CICT_PRIMARY_BUILDING_FLOORS.get(building)
            if allowed is None:
                return False
            return floor in allowed

        return False

    def _is_campus_place_node(self, node: Dict) -> bool:
        name = str(node.get("name") or "").strip().lower()
        kind = str(node.get("kind") or "").strip().lower()
        if kind in {"campus_place", "building", "landmark", "gate"}:
            return True
        if kind in {"room", "lab", "office", "stairs"}:
            return False
        return any(tok in name for tok in CAMPUS_PLACE_TOKENS)

    def _place_type(self, node: Dict) -> str:
        if self._is_cict_indoor_node(node):
            return "cict_indoor"
        if self._is_campus_place_node(node):
            return "campus_place"
        return "other"

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
                if not self._edge_allowed(cur, nxt):
                    continue
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

    @staticmethod
    def _as_float(value) -> Optional[float]:
        try:
            return float(value)
        except Exception:
            return None

    def _node_xy(self, node_id: str) -> Optional[Tuple[float, float]]:
        node = self._nodes.get(node_id, {})
        x = self._as_float(node.get("x"))
        y = self._as_float(node.get("y"))
        if x is None or y is None:
            return None
        return (x, y)

    def _heuristic(self, node_id: str, goal_id: str) -> float:
        a = self._node_xy(node_id)
        b = self._node_xy(goal_id)
        if not a or not b:
            return 0.0

        node_a = self._nodes.get(node_id, {})
        node_b = self._nodes.get(goal_id, {})
        same_building = str(node_a.get("building") or "").strip().lower() == str(node_b.get("building") or "").strip().lower()
        same_floor = str(node_a.get("floor") or "").strip().lower() == str(node_b.get("floor") or "").strip().lower()
        if not (same_building and same_floor):
            return 0.0

        return math.hypot(a[0] - b[0], a[1] - b[1])

    def shortest_path_astar(self, start_id: str, goal_id: str) -> Optional[Dict]:
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
        g_score = {start_id: 0.0}
        prev: Dict[str, Optional[str]] = {start_id: None}
        heap: List[Tuple[float, str]] = [(self._heuristic(start_id, goal_id), start_id)]

        while heap:
            _, cur = heappop(heap)
            if cur == goal_id:
                break

            cur_g = g_score.get(cur, math.inf)
            if cur_g == math.inf:
                continue

            for nxt, weight in adjacency.get(cur, []):
                if not self._edge_allowed(cur, nxt):
                    continue
                candidate_g = cur_g + weight
                if candidate_g < g_score.get(nxt, math.inf):
                    g_score[nxt] = candidate_g
                    prev[nxt] = cur
                    f_score = candidate_g + self._heuristic(nxt, goal_id)
                    heappush(heap, (f_score, nxt))

        if goal_id not in g_score:
            return None

        path_ids = []
        cursor = goal_id
        while cursor is not None:
            path_ids.append(cursor)
            cursor = prev.get(cursor)
        path_ids.reverse()

        return {
            "distance": round(g_score[goal_id], 2),
            "path_ids": path_ids,
            "path_names": [self._nodes[n].get("name", n) for n in path_ids],
        }

    def compute_route(self, start_hint: str, target_hint: str, *, algorithm: str = "astar") -> Optional[Dict]:
        start_id = self._match_node_id(start_hint)
        target_id = self._match_node_id(target_hint)
        if not start_id or not target_id:
            return None

        return self.compute_route_by_ids(start_id, target_id, algorithm=algorithm)

    def compute_route_by_ids(self, start_id: str, target_id: str, *, algorithm: str = "astar") -> Optional[Dict]:
        if start_id not in self._nodes or target_id not in self._nodes:
            return None

        algo = (algorithm or "astar").strip().lower()
        if algo == "dijkstra":
            route = self.shortest_path(start_id, target_id)
            algorithm_used = "dijkstra"
        else:
            route = self.shortest_path_astar(start_id, target_id)
            algorithm_used = "astar"

        if not route:
            return None

        return self._build_route_payload(start_id, target_id, route, algorithm_used)

    def _build_route_payload(self, start_id: str, target_id: str, route: Dict, algorithm_used: str) -> Dict:
        points = []
        for node_id in route.get("path_ids", []):
            xy = self._node_xy(node_id)
            node = self._nodes.get(node_id, {})
            points.append(
                {
                    "id": node_id,
                    "name": node.get("name", node_id),
                    "x": xy[0] if xy else None,
                    "y": xy[1] if xy else None,
                    "building": node.get("building", "Unknown Building"),
                    "floor": node.get("floor", "Unknown Floor"),
                }
            )

        directions = self._directions_from_points(points)

        start_node = self._nodes.get(start_id, {})
        target_node = self._nodes.get(target_id, {})
        return {
            "algorithm": algorithm_used,
            "distance": route.get("distance", 0.0),
            "path_ids": route.get("path_ids", []),
            "path_names": route.get("path_names", []),
            "points": points,
            "directions": directions,
            "directions_text": "\n".join(f"{idx}. {step}" for idx, step in enumerate(directions, start=1)),
            "start": {
                "id": start_id,
                "name": start_node.get("name", start_id),
                "building": start_node.get("building", "Unknown Building"),
                "floor": start_node.get("floor", "Unknown Floor"),
            },
            "target": {
                "id": target_id,
                "name": target_node.get("name", target_id),
                "building": target_node.get("building", "Unknown Building"),
                "floor": target_node.get("floor", "Unknown Floor"),
            },
        }

    @staticmethod
    def _directions_from_points(points: List[Dict]) -> List[str]:
        if not points:
            return []

        directions: List[str] = []
        first = points[0]
        directions.append(
            f"Start at {first.get('name', 'the start point')} "
            f"({first.get('building', 'Unknown Building')}, {first.get('floor', 'Unknown Floor')})."
        )

        for idx in range(1, len(points)):
            prev = points[idx - 1]
            cur = points[idx]
            prev_building = str(prev.get("building") or "Unknown Building")
            cur_building = str(cur.get("building") or "Unknown Building")
            prev_floor = str(prev.get("floor") or "Unknown Floor")
            cur_floor = str(cur.get("floor") or "Unknown Floor")
            cur_name = str(cur.get("name") or "the next point")

            if prev_building != cur_building:
                directions.append(
                    f"Move from {prev_building} to {cur_building}, then proceed to {cur_name}."
                )
            elif prev_floor != cur_floor:
                directions.append(
                    f"Change floor from {prev_floor} to {cur_floor}, then proceed to {cur_name}."
                )
            else:
                directions.append(f"Proceed to {cur_name}.")

        if len(points) > 1:
            directions.append(f"You have arrived at {points[-1].get('name', 'your destination')}.")

        return directions

    def route_options(self) -> List[Dict]:
        adjacency = self._adjacency()

        component_of: Dict[str, int] = {}
        comp_idx = 0
        for node_id in self._nodes.keys():
            if node_id in component_of:
                continue
            stack = [node_id]
            while stack:
                cur = stack.pop()
                if cur in component_of:
                    continue
                component_of[cur] = comp_idx
                for nxt, _ in adjacency.get(cur, []):
                    if nxt not in component_of:
                        stack.append(nxt)
            comp_idx += 1

        options = []
        seen = set()
        for node_id, node in self._nodes.items():
            raw_name = str(node.get("name") or "").strip()
            name = self._clean_bullet_item(raw_name)
            if not name:
                continue
            if len(name) > 70:
                continue

            place_type = self._place_type(node)
            if place_type not in {"cict_indoor", "campus_place"}:
                continue

            key = (name.lower(), str(node.get("building") or "").lower(), str(node.get("floor") or "").lower())
            if key in seen:
                continue
            seen.add(key)
            building = str(node.get("building") or "Unknown Building").strip()
            floor = str(node.get("floor") or "Unknown Floor").strip()
            xy = self._node_xy(node_id)
            options.append(
                {
                    "id": node_id,
                    "name": name,
                    "building": building,
                    "floor": floor,
                    "place_type": place_type,
                    "component": component_of.get(node_id, -1),
                    "has_coordinates": bool(xy),
                    "label": f"[{ 'Campus' if place_type == 'campus_place' else 'CICT' }] {name} ({building}, {floor})",
                }
            )

        options.sort(key=lambda item: (item["building"], item["floor"], item["name"]))
        return options

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
