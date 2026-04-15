import os
import json
import logging
import re
from typing import List, Dict

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
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS as FAISSLocal
except Exception:
    HuggingFaceEmbeddings = None  # type: ignore[assignment]
    FAISSLocal = None  # type: ignore[assignment]

from .config import CONFIG, FAISS_DIR, FAISS_MANIFEST_PATH, pdf_paths

try:
    import PyPDF2
except Exception:
    PyPDF2 = None


class RAGStore:
    def __init__(self) -> None:
        # Keep startup logs clean while preserving runtime behavior.
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

        self._embeddings = None
        if HuggingFaceEmbeddings is not None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        self._vectorstore = None

    def _extract_pdf_text(self, path: str) -> str:
        if PyPDF2 is None:
            return ""
        try:
            reader = PyPDF2.PdfReader(path)
            chunks: List[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    chunks.append(self._clean_text(text))
            return "\n".join(chunks)
        except Exception:
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = (text or "").replace("\u00ad", "")
        cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _source_boost(question: str, source: str) -> int:
        q = (question or "").lower()
        s = (source or "").lower()
        boost = 0

        if ("dean" in q or "coordinator" in q or "director" in q) and "cictify - faqs" in s:
            boost += 5
        if ("transferee" in q or "shiftee" in q or "returnee" in q or "admission" in q) and "enhanced-guidelines" in s:
            boost += 6
        if ("acad" in q or "room" in q or "floor" in q or "lab" in q or "where" in q) and ("cict rooms" in s or "cict-rooms" in s):
            boost += 5
        if "dean" in q and "name" in q and "student handbook" in s:
            boost -= 2

        return boost

    def _manifest_payload(self) -> Dict:
        return {
            "sources": sorted(pdf_paths()),
            "chunk_size": CONFIG.chunk_size,
            "chunk_overlap": CONFIG.chunk_overlap,
        }

    def _manifest_matches(self) -> bool:
        if not FAISS_MANIFEST_PATH.exists():
            return False
        try:
            existing = json.loads(FAISS_MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return False
        return existing == self._manifest_payload()

    def _save_manifest(self) -> None:
        FAISS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAISS_MANIFEST_PATH.write_text(json.dumps(self._manifest_payload(), ensure_ascii=True, indent=2), encoding="utf-8")

    def load_or_build(self) -> None:
        if FAISSLocal is None or self._embeddings is None:
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        )

        faiss_dir = str(FAISS_DIR)
        if os.path.exists(faiss_dir) and os.path.exists(f"{faiss_dir}.faiss") and self._manifest_matches():
            self._vectorstore = FAISSLocal.load_local(
                faiss_dir,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        all_texts: List[str] = []
        all_meta: List[Dict] = []
        for path in pdf_paths():
            if not os.path.exists(path):
                continue
            text = self._extract_pdf_text(path)
            if not text.strip():
                continue
            for chunk in text_splitter.split_text(text):
                all_texts.append(chunk)
                all_meta.append({"source_file": os.path.basename(path)})

        if not all_texts:
            return

        self._vectorstore = FAISSLocal.from_texts(
            texts=all_texts,
            embedding=self._embeddings,
            metadatas=all_meta,
        )
        os.makedirs(os.path.dirname(faiss_dir), exist_ok=True)
        self._vectorstore.save_local(faiss_dir)
        self._save_manifest()

    def query(self, question: str, k: int | None = None) -> List[Dict]:
        if not self._vectorstore:
            return []

        def query_terms(text: str) -> List[str]:
            stop = {
                "the", "and", "for", "with", "that", "this", "from", "have", "what", "where", "when",
                "which", "who", "why", "how", "can", "you", "your", "are", "is", "to", "of", "in",
                "ang", "mga", "ano", "saan", "sino", "paano", "pwede", "ba", "po", "ng", "sa", "at",
            }
            terms = re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())
            return [t for t in terms if t not in stop]

        retriever = self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or CONFIG.retrieval_k},
        )
        docs = retriever.invoke(question)

        q_lower = (question or "").lower()
        extra_queries: List[str] = []
        if "dean" in q_lower and ("who" in q_lower or "name" in q_lower):
            extra_queries.append("BulSU CICT Administration Dean CICT")
        if "transferee" in q_lower or "shiftee" in q_lower or "returnee" in q_lower:
            extra_queries.append("Definition of terms transferee shiftee returnee BulSU")
        if "acad" in q_lower or ("where" in q_lower and ("room" in q_lower or "lab" in q_lower)):
            extra_queries.append("CICT Rooms floor plan Acad 1 location")

        for hint in extra_queries:
            try:
                docs.extend(retriever.invoke(hint))
            except Exception:
                continue

        rows = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source_file", "Unknown"),
            }
            for doc in docs
        ]

        # Remove near duplicates early to reduce repetitive fallback answers.
        deduped_rows = []
        seen = set()
        for row in rows:
            key = re.sub(r"\s+", " ", (row.get("content") or "").strip().lower())[:500]
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped_rows.append(row)
        rows = deduped_rows

        q_terms = query_terms(question)
        if not q_terms:
            return rows[: CONFIG.rag_rerank_top_n]

        scored = []
        for row in rows:
            text = (row.get("content") or "").lower()
            score = (sum(1 for t in q_terms if t in text) * 3) + self._source_boost(question, row.get("source", ""))
            scored.append((score, row))

        filtered = [r for s, r in scored if s >= CONFIG.rag_min_term_overlap]
        if not filtered:
            filtered = [r for _, r in scored[: max(2, CONFIG.rag_rerank_top_n)]]
        else:
            filtered = [r for _, r in sorted(scored, key=lambda item: item[0], reverse=True) if r in filtered]

        return filtered[: CONFIG.rag_rerank_top_n]

    def context_for(self, question: str) -> str:
        docs = self.query(question)
        if not docs:
            return ""
        blocks = [f"Source: {d['source']}\n{d['content']}" for d in docs]
        return "\n\n---\n\n".join(blocks)
