import json
import re
from datetime import datetime
from typing import Dict, List

from .config import FLOORPLAN_CONTEXT_PATH


class FloorplanContextStore:
    def __init__(self) -> None:
        self._path = FLOORPLAN_CONTEXT_PATH
        self._records: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._records = []
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = payload.get("records", []) if isinstance(payload, dict) else []
        except Exception:
            self._records = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": datetime.now().isoformat(), "records": self._records}
        self._path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def add_record(self, *, title: str, building: str, floor: str, ocr_text: str, source_file: str) -> Dict:
        record = {
            "title": title.strip() or "Floorplan",
            "building": building.strip() or "Unknown Building",
            "floor": floor.strip() or "Unknown Floor",
            "ocr_text": (ocr_text or "").strip(),
            "source_file": source_file,
            "timestamp": datetime.now().isoformat(),
        }
        self._records.append(record)
        self._save()
        return record

    @staticmethod
    def _terms(query: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Z0-9]{3,}", (query or "").lower())]

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        terms = self._terms(query)
        if not self._records:
            return []
        if not terms:
            return self._records[-top_k:]

        scored = []
        for rec in self._records:
            hay = f"{rec.get('title','')} {rec.get('building','')} {rec.get('floor','')} {rec.get('ocr_text','')}".lower()
            score = sum(hay.count(term) for term in terms)
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [rec for score, rec in scored if score > 0][:top_k]
        return selected or [rec for _, rec in scored[:top_k]]

    def context_for(self, query: str) -> str:
        hits = self.search(query)
        if not hits:
            return ""
        blocks = []
        for hit in hits:
            blocks.append(
                "\n".join(
                    [
                        f"Title: {hit.get('title','')}",
                        f"Building: {hit.get('building','')}",
                        f"Floor: {hit.get('floor','')}",
                        f"Source: {hit.get('source_file','')}",
                        f"OCR: {hit.get('ocr_text','')}",
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)
