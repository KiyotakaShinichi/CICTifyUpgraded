import asyncio
from typing import Optional

from .floorplan_context import FloorplanContextStore
from .retrieval import RAGStore

try:
    from vision_reader import read_image_with_groq
except Exception:
    read_image_with_groq = None


class OCRIngestion:
    def __init__(self, rag_store: RAGStore, floorplan_store: FloorplanContextStore | None = None) -> None:
        self._rag_store = rag_store
        self._floorplan_store = floorplan_store

    async def extract_text(self, image_bytes: bytes) -> Optional[str]:
        if read_image_with_groq is None:
            return None
        return await read_image_with_groq(
            image_bytes,
            question="Extract text only. Preserve key terms exactly.",
        )

    def ingest_text(self, text: str) -> None:
        # Placeholder hook for future incremental FAISS updates.
        # Existing index build remains deterministic from documents.
        _ = text

    async def ingest_image(self, image_bytes: bytes) -> Optional[str]:
        text = await self.extract_text(image_bytes)
        if text:
            self.ingest_text(text)
        return text

    async def ingest_floorplan(self, image_bytes: bytes, *, title: str, building: str, floor: str, source_file: str) -> Optional[dict]:
        text = await self.extract_text(image_bytes)
        if not text or self._floorplan_store is None:
            return None
        return self._floorplan_store.add_record(
            title=title,
            building=building,
            floor=floor,
            ocr_text=text,
            source_file=source_file,
        )
