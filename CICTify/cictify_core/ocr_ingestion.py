import asyncio
from typing import Optional

from .floorplan_context import FloorplanContextStore
from .retrieval import RAGStore
from .spatial_graph import SpatialGraphStore

try:
    from vision_reader import (
        classify_image_with_groq,
        extract_floorplan_graph_with_groq,
        extract_room_status_with_groq,
        read_image_with_groq,
    )
except Exception:
    read_image_with_groq = None
    classify_image_with_groq = None
    extract_floorplan_graph_with_groq = None
    extract_room_status_with_groq = None


class OCRIngestion:
    def __init__(
        self,
        rag_store: RAGStore,
        floorplan_store: FloorplanContextStore | None = None,
        spatial_graph_store: SpatialGraphStore | None = None,
    ) -> None:
        self._rag_store = rag_store
        self._floorplan_store = floorplan_store
        self._spatial_graph_store = spatial_graph_store

    async def extract_text(self, image_bytes: bytes) -> Optional[str]:
        if read_image_with_groq is None:
            return None
        return await read_image_with_groq(
            image_bytes,
            question="Extract text only. Preserve key terms exactly.",
        )

    @staticmethod
    def vision_available() -> bool:
        return read_image_with_groq is not None

    def ingest_text(self, text: str) -> None:
        # Placeholder hook for future incremental FAISS updates.
        # Existing index build remains deterministic from documents.
        _ = text

    async def ingest_image(self, image_bytes: bytes) -> Optional[str]:
        text = await self.extract_text(image_bytes)
        if text:
            self.ingest_text(text)
        return text

    async def classify_image(self, image_bytes: bytes) -> dict:
        if classify_image_with_groq is None:
            return {"type": "other", "confidence": 0.0, "reason": "classifier_unavailable"}
        result = await classify_image_with_groq(image_bytes)
        if not isinstance(result, dict):
            return {"type": "other", "confidence": 0.0, "reason": "invalid_classifier_output"}
        return result

    async def ingest_floorplan(self, image_bytes: bytes, *, title: str, building: str, floor: str, source_file: str) -> Optional[dict]:
        text = await self.extract_text(image_bytes)
        if not text or self._floorplan_store is None:
            return None

        record = self._floorplan_store.add_record(
            title=title,
            building=building,
            floor=floor,
            ocr_text=text,
            source_file=source_file,
        )

        graph_stats = None
        if extract_floorplan_graph_with_groq is not None and self._spatial_graph_store is not None:
            graph = await extract_floorplan_graph_with_groq(image_bytes, building=building, floor=floor)
            graph_stats = self._spatial_graph_store.add_graph(graph, source_file=source_file)

        return {
            **record,
            "spatial_graph": graph_stats,
        }

    async def ingest_image_auto(self, image_bytes: bytes, *, title: str, building: str, floor: str, source_file: str) -> dict:
        image_type = await self.classify_image(image_bytes)
        result = {
            "image_type": image_type,
            "record": None,
            "ocr_text": None,
            "status_update": None,
        }

        if image_type.get("type") == "floor_plan":
            result["record"] = await self.ingest_floorplan(
                image_bytes,
                title=title,
                building=building,
                floor=floor,
                source_file=source_file,
            )
        else:
            result["ocr_text"] = await self.ingest_image(image_bytes)
            if self._spatial_graph_store is not None and extract_room_status_with_groq is not None:
                status = await extract_room_status_with_groq(
                    image_bytes,
                    default_building=building,
                    default_floor=floor,
                )
                if status.get("room_name") and float(status.get("confidence", 0.0)) >= 0.5:
                    has_open_pc = bool(status.get("has_open_pc", False))
                    if bool(status.get("is_lab", False)):
                        has_open_pc = has_open_pc
                    result["status_update"] = self._spatial_graph_store.upsert_room_status(
                        room_name=str(status.get("room_name") or "").strip(),
                        building=str(status.get("building") or building).strip() or building,
                        floor=str(status.get("floor") or floor).strip() or floor,
                        has_open_pc=has_open_pc,
                        source_file=source_file,
                    )

        return result
