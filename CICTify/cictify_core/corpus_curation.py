import hashlib
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List

from .config import CONFIG, CURATED_CORPUS_PATH, KB_CURATION_REPORT_PATH

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        class RecursiveCharacterTextSplitter:  # type: ignore[override]
            def __init__(self, chunk_size: int, chunk_overlap: int, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                chunks: List[str] = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunks.append(text[start:end])
                    if end >= len(text):
                        break
                    start = max(0, end - self.chunk_overlap)
                return chunks

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]
    util = None  # type: ignore[assignment]


@dataclass
class CuratedChunk:
    content: str
    source_file: str
    title: str
    doc_hash: str
    chunk_hash: str
    chunk_index: int
    section_index: int


@dataclass
class CuratedDocument:
    source_file: str
    title: str
    raw_text: str
    normalized_text: str
    text_hash: str
    char_count: int
    word_count: int


class CorpusCurator:
    def __init__(self, source_paths: Iterable[str]) -> None:
        self.source_paths = [str(Path(path)) for path in source_paths]
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        )
        self._semantic_model = None
        if CONFIG.semantic_chunking and SentenceTransformer is not None:
            try:
                self._semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self._semantic_model = None

    @staticmethod
    def normalize_text(text: str) -> str:
        cleaned = (text or "").replace("\u00ad", "")
        cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.replace("•", "\n• ")
        cleaned = cleaned.replace("●", "\n● ")
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        return cleaned.strip()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _title_from_path(path: str) -> str:
        return Path(path).stem.replace("_", " ").strip()

    def _extract_pdf_text(self, path: str) -> str:
        if PyPDF2 is None:
            return ""
        try:
            reader = PyPDF2.PdfReader(path)
            pages: List[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            return "\n".join(pages)
        except Exception:
            return ""

    def load_documents(self) -> List[CuratedDocument]:
        seen_hashes = set()
        documents: List[CuratedDocument] = []

        for path_str in self.source_paths:
            path = Path(path_str)
            if not path.exists() or path.stat().st_size == 0:
                continue

            raw_text = self._extract_pdf_text(path_str)
            normalized_text = self.normalize_text(raw_text)
            if not normalized_text:
                continue

            text_hash = self._hash(normalized_text)
            if text_hash in seen_hashes:
                continue

            # Near-duplicate guard for documents with similar OCR output.
            if any(self._similarity(normalized_text, doc.normalized_text) >= 0.985 for doc in documents):
                continue

            seen_hashes.add(text_hash)
            documents.append(
                CuratedDocument(
                    source_file=str(path),
                    title=self._title_from_path(path_str),
                    raw_text=raw_text,
                    normalized_text=normalized_text,
                    text_hash=text_hash,
                    char_count=len(normalized_text),
                    word_count=len(re.findall(r"[A-Za-z0-9']+", normalized_text)),
                )
            )

        return documents

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a[:6000], b[:6000]).ratio()

    @staticmethod
    def _dedupe_chunks(chunks: List[CuratedChunk]) -> List[CuratedChunk]:
        seen = set()
        deduped: List[CuratedChunk] = []
        for chunk in chunks:
            key = re.sub(r"\s+", " ", chunk.content).strip().lower()[:700]
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped

    def build_corpus(self) -> Dict:
        documents = self.load_documents()
        chunks: List[CuratedChunk] = []
        dropped_sources = len(self.source_paths) - len(documents)

        for doc in documents:
            sections = [s.strip() for s in re.split(r"\n\s*\n|(?<=\.)\s+(?=[A-Z0-9●•])", doc.normalized_text) if s.strip()]
            if not sections:
                sections = [doc.normalized_text]

            for section_index, section in enumerate(sections):
                for chunk_index, chunk_text in enumerate(self._split_section(section)):
                    cleaned_chunk = self.normalize_text(chunk_text)
                    if len(cleaned_chunk) < 80:
                        continue
                    chunk_hash = self._hash(f"{doc.text_hash}:{cleaned_chunk[:1000]}")
                    chunks.append(
                        CuratedChunk(
                            content=cleaned_chunk,
                            source_file=doc.source_file,
                            title=doc.title,
                            doc_hash=doc.text_hash,
                            chunk_hash=chunk_hash,
                            chunk_index=chunk_index,
                            section_index=section_index,
                        )
                    )

        chunks = self._dedupe_chunks(chunks)
        report = self._build_report(documents, chunks, dropped_sources)
        return {
            "documents": [asdict(doc) for doc in documents],
            "chunks": [asdict(chunk) for chunk in chunks],
            "report": report,
        }

    def _split_section(self, section: str) -> List[str]:
        if not section.strip():
            return []

        if not CONFIG.semantic_chunking or self._semantic_model is None or util is None:
            return self._splitter.split_text(section)

        try:
            return self._semantic_chunks(section)
        except Exception:
            return self._splitter.split_text(section)

    def _semantic_chunks(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) <= 2:
            return self._splitter.split_text(text)

        vectors = self._semantic_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        chunks: List[str] = []
        current: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            prev_vec = vectors[i - 1]
            cur_vec = vectors[i]
            sim = float(util.cos_sim(prev_vec, cur_vec).item())

            candidate = " ".join(current + [sentences[i]])
            too_long = len(candidate) >= CONFIG.chunk_size
            semantic_break = sim < CONFIG.semantic_break_threshold

            if too_long or (semantic_break and len(" ".join(current)) >= CONFIG.semantic_min_chunk_chars):
                chunks.append(" ".join(current).strip())
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            chunks.append(" ".join(current).strip())

        expanded: List[str] = []
        for chunk in chunks:
            if len(chunk) > CONFIG.chunk_size * 1.4:
                expanded.extend(self._splitter.split_text(chunk))
            else:
                expanded.append(chunk)
        return [c for c in expanded if c.strip()]

    @staticmethod
    def _build_report(documents: List[CuratedDocument], chunks: List[CuratedChunk], dropped_sources: int) -> Dict:
        by_source: Dict[str, int] = {}
        for chunk in chunks:
            by_source[chunk.source_file] = by_source.get(chunk.source_file, 0) + 1

        return {
            "source_total": len(documents) + dropped_sources,
            "source_kept": len(documents),
            "source_dropped": dropped_sources,
            "chunk_total": len(chunks),
            "chunk_size": CONFIG.chunk_size,
            "chunk_overlap": CONFIG.chunk_overlap,
            "semantic_chunking": CONFIG.semantic_chunking,
            "semantic_break_threshold": CONFIG.semantic_break_threshold,
            "top_sources": sorted(by_source.items(), key=lambda item: item[1], reverse=True)[:10],
        }

    def save_cache(self, payload: Dict) -> None:
        CURATED_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CURATED_CORPUS_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def save_report(self, report: Dict) -> None:
        KB_CURATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# KB Curation Report",
            "",
            f"- Source PDFs scanned: {report.get('source_total', 0)}",
            f"- Sources kept: {report.get('source_kept', 0)}",
            f"- Sources dropped: {report.get('source_dropped', 0)}",
            f"- Final chunks: {report.get('chunk_total', 0)}",
            f"- Chunk size: {report.get('chunk_size', CONFIG.chunk_size)}",
            f"- Chunk overlap: {report.get('chunk_overlap', CONFIG.chunk_overlap)}",
            f"- Semantic chunking: {report.get('semantic_chunking', CONFIG.semantic_chunking)}",
            f"- Semantic break threshold: {report.get('semantic_break_threshold', CONFIG.semantic_break_threshold)}",
            "",
            "## Top Chunk Sources",
        ]
        for source, count in report.get("top_sources", []):
            lines.append(f"- {Path(source).name}: {count} chunks")
        KB_CURATION_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def curate_corpus(source_paths: Iterable[str]) -> Dict:
    curator = CorpusCurator(source_paths)
    payload = curator.build_corpus()
    curator.save_cache(payload)
    curator.save_report(payload["report"])
    return payload
