import re
from typing import Optional

from .config import CONFIG
from .spatial_graph import SpatialGraphStore


class SpatialAwarenessTool:
    """Intent-gated wrapper over spatial graph reasoning.

    The goal is to keep spatial handling isolated and non-intrusive to the
    main General -> RAG -> Web pipeline.
    """

    def __init__(self, graph_store: SpatialGraphStore) -> None:
        self.graph_store = graph_store

    @staticmethod
    def _has_room_signal(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False

        keywords = {
            "room",
            "hall",
            "building",
            "campus",
            "landmark",
            "gate",
            "floor",
            "stairs",
            "staircase",
            "lab",
            "office",
            "dean",
            "faculty",
            "networking",
            "server",
            "avr",
            "ideation",
            "conference",
            "pimentel",
            "nstp",
            "acad",
            "ct",
            "sdl",
            "it",
        }
        if any(word in t for word in keywords):
            return True

        # Matches short room codes such as A1, IT14, CT6, SDL2, Acad 7.
        return bool(re.search(r"\b([a-z]{1,8}\s?\d{1,3}|[a-z]\d{1,3})\b", t, flags=re.IGNORECASE))

    @staticmethod
    def _has_spatial_intent(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False

        spatial_terms = [
            "near",
            "nearby",
            "beside",
            "next to",
            "between",
            "left",
            "right",
            "where is",
            "how do i get",
            "how to get",
            "route",
            "path",
            "directions",
            "closest",
            "nearest",
            "nasaan",
            "nasan",
            "asan",
            "malapit",
            "katabi",
            "papunta",
            "paano pumunta",
            "which part of the campus",
            "what part of the campus",
            "which part",
            "located",
            "location",
        ]
        return any(term in t for term in spatial_terms)

    def should_run(self, question: str) -> bool:
        if not CONFIG.spatial_tool_enabled:
            return False

        if not CONFIG.spatial_tool_strict_intent:
            return self._has_room_signal(question)

        return self._has_spatial_intent(question) and self._has_room_signal(question)

    def run(self, question: str) -> Optional[str]:
        if not self.should_run(question):
            return None
        return self.graph_store.answer_navigation_query(question)
