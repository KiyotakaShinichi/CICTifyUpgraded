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

from .config import CONFIG, CURATED_CORPUS_PATH, FAISS_DIR, FAISS_MANIFEST_PATH, pdf_paths
from .corpus_curation import curate_corpus

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

    @staticmethod
    def _source_family(question: str) -> List[str]:
        q = (question or "").lower()
        if any(term in q for term in ["dean", "coordinator", "director", "who is", "name"]):
            return ["cictify - faqs", "faculty manual for bor"]
        if any(term in q for term in ["shiftee", "transferee", "returnee", "loa", "admission", "change program"]):
            return ["bulsu-enhanced-guidelines", "bulsu student handbook"]
        if any(term in q for term in ["acad", "room", "floor", "lab", "where is", "where", "location"]):
            return ["cict rooms - pics & desc", "cict-rooms"]
        if any(term in q for term in ["mission", "vision", "programs offered", "tracks", "bsit", "bsis", "blis"]):
            return ["cictify - faqs", "bulsu student handbook"]
        return []

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

    def _load_curated_cache(self) -> List[Dict]:
        if not CURATED_CORPUS_PATH.exists() or not self._manifest_matches():
            curated = curate_corpus(pdf_paths())
            return curated.get("chunks", [])

        try:
            payload = json.loads(CURATED_CORPUS_PATH.read_text(encoding="utf-8"))
            chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
            return chunks if isinstance(chunks, list) else []
        except Exception:
            curated = curate_corpus(pdf_paths())
            return curated.get("chunks", [])

    def load_or_build(self) -> None:
        if FAISSLocal is None or self._embeddings is None:
            return

        faiss_dir = str(FAISS_DIR)
        if os.path.exists(faiss_dir) and os.path.exists(f"{faiss_dir}.faiss") and self._manifest_matches():
            self._vectorstore = FAISSLocal.load_local(
                faiss_dir,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        curated_chunks = self._load_curated_cache()

        all_texts: List[str] = []
        all_meta: List[Dict] = []
        for chunk in curated_chunks:
            content = (chunk or {}).get("content", "")
            source_file = (chunk or {}).get("source_file", "Unknown")
            if not content:
                continue
            all_texts.append(content)
            all_meta.append(
                {
                    "source_file": os.path.basename(source_file),
                    "doc_hash": (chunk or {}).get("doc_hash", ""),
                    "chunk_hash": (chunk or {}).get("chunk_hash", ""),
                    "section_index": (chunk or {}).get("section_index", 0),
                    "chunk_index": (chunk or {}).get("chunk_index", 0),
                }
            )

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

        def role_seed_rows(text: str, limit: int = 4) -> List[Dict]:
            q = (text or "").lower()
            if not re.search(r"\b(?:who\s+is|sino\s+si|sino\s+ang)\b", q):
                return []

            role_patterns = []
            if "associate dean" in q or "assoc dean" in q:
                role_patterns.append(r"\bassociate\s+dean\b")
            if "dean" in q:
                role_patterns.append(r"\bdean\b")
            if "program chair" in q:
                role_patterns.append(r"\bprogram\s+chair\b")
            if not role_patterns:
                return []

            chunks = self._load_curated_cache()
            seeded: List[Dict] = []
            seen = set()
            doc_seeded: List[Dict] = []
            for chunk in chunks:
                content = (chunk or {}).get("content", "")
                if not content:
                    continue
                normalized = re.sub(r"\s+", " ", content).strip()
                nlower = normalized.lower()
                if "cict" not in nlower:
                    continue
                if not any(re.search(pat, nlower) for pat in role_patterns):
                    continue
                key = nlower[:500]
                if key in seen:
                    continue
                seen.add(key)
                source_file = os.path.basename((chunk or {}).get("source_file", "Unknown"))
                seeded.append({"content": normalized, "source": source_file})

            # Some role/name entries are more complete in document-level
            # normalized_text than in chunked snippets. Add those as seeds too.
            if CURATED_CORPUS_PATH.exists():
                try:
                    payload = json.loads(CURATED_CORPUS_PATH.read_text(encoding="utf-8"))
                    docs = payload.get("documents", []) if isinstance(payload, dict) else []
                except Exception:
                    docs = []

                for doc in docs:
                    content = (doc or {}).get("normalized_text", "")
                    if not content:
                        continue
                    normalized = re.sub(r"\s+", " ", content).strip()
                    nlower = normalized.lower()
                    if "cict" not in nlower:
                        continue
                    if not any(re.search(pat, nlower) for pat in role_patterns):
                        continue
                    # Prefer rows that include likely person-name markers.
                    if not re.search(r"\b(?:dr|mr|ms|engr)\.\s+[a-z]", nlower):
                        continue
                    key = nlower[:500]
                    if key in seen:
                        continue
                    seen.add(key)
                    source_file = os.path.basename((doc or {}).get("source_file", "Unknown"))
                    doc_seeded.append({"content": normalized, "source": source_file})

            # Prefer seeds that include complete honorific+name patterns.
            def _seed_rank(row: Dict) -> int:
                txt = (row.get("content") or "")
                score = 0
                if re.search(r"\b(?:Dr|Mr|Ms|Engr)\.?\s+[A-Z]", txt):
                    score += 4
                if re.search(r"\bdean\s*,?\s*cict\b", txt, flags=re.IGNORECASE):
                    score += 3
                if re.search(r"\bassociate\s+dean\s*,?\s*cict\b", txt, flags=re.IGNORECASE):
                    score += 2
                return score

            ordered = sorted(doc_seeded, key=_seed_rank, reverse=True) + sorted(seeded, key=_seed_rank, reverse=True)
            return ordered[:limit]

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

        # Seed role/name queries directly from curated chunks to avoid missing
        # administration entries when embedding search returns truncated variants.
        rows = role_seed_rows(question) + rows

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

        family_terms = self._source_family(question)
        if family_terms:
            prioritized = [row for row in rows if any(term in (row.get("source", "").lower()) for term in family_terms)]
            if prioritized:
                rows = prioritized

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
