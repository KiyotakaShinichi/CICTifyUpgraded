import os
import pathlib
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
GUI_DIR = BASE_DIR / "gui"
FAISS_DIR = BASE_DIR / "vectorstore" / "faiss_index"
FAISS_MANIFEST_PATH = BASE_DIR / "vectorstore" / "faiss_manifest.json"
CURATED_CORPUS_PATH = BASE_DIR / "vectorstore" / "curated_corpus.json"
DB_PATH = BASE_DIR / "vectorstore" / "cictify.db"
KB_PDF_DIR = BASE_DIR / "knowledge_base" / "pdfs"
FLOORPLAN_CONTEXT_PATH = BASE_DIR / "vectorstore" / "floorplan_context.json"
SPATIAL_GRAPH_PATH = BASE_DIR / "vectorstore" / "spatial_graph.json"
KB_CURATION_REPORT_PATH = BASE_DIR / "evaluation" / "kb_curation_report.md"

# Load secrets/config from local .env file if present.
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    app_name: str = os.getenv("APP_NAME", "CICTify")
    routing_model: str = os.getenv("ROUTING_MODEL", "llama-3.3-70b-versatile")
    chat_models: tuple[str, ...] = (
        os.getenv("ANSWER_MODEL", "openai/gpt-oss-120b"),
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    # ===== HYPERPARAMETERS =====
    # Uppercase aliases are kept for backward compatibility with older code paths.
    CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 1500)
    CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 400)
    MAX_TOKENS: int = _env_int("MAX_TOKENS", 2000)
    TEMPERATURE: float = _env_float("TEMPERATURE", 1.0)
    RETRIEVAL_K: int = _env_int("RETRIEVAL_K", 12)

    RAG_RERANK_TOP_N: int = _env_int("RAG_RERANK_TOP_N", 5)
    RAG_MIN_TERM_OVERLAP: int = _env_int("RAG_MIN_TERM_OVERLAP", 2)
    MEMORY_TOP_K: int = _env_int("MEMORY_TOP_K", 4)
    MEMORY_MIN_SCORE: int = _env_int("MEMORY_MIN_SCORE", 1)

    SEMANTIC_CHUNKING: bool = _env_bool("SEMANTIC_CHUNKING", True)
    SEMANTIC_BREAK_THRESHOLD: float = _env_float("SEMANTIC_BREAK_THRESHOLD", 0.45)
    SEMANTIC_MIN_CHUNK_CHARS: int = _env_int("SEMANTIC_MIN_CHUNK_CHARS", 220)

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    retrieval_k: int = RETRIEVAL_K
    rag_rerank_top_n: int = RAG_RERANK_TOP_N
    rag_min_term_overlap: int = RAG_MIN_TERM_OVERLAP
    memory_top_k: int = MEMORY_TOP_K
    memory_min_score: int = MEMORY_MIN_SCORE
    semantic_chunking: bool = SEMANTIC_CHUNKING
    semantic_break_threshold: float = SEMANTIC_BREAK_THRESHOLD
    semantic_min_chunk_chars: int = SEMANTIC_MIN_CHUNK_CHARS

    max_history_messages: int = 20
    max_memory_messages: int = 200
    request_timeout_sec: int = _env_int("REQUEST_TIMEOUT_SEC", 25)
    quick_timeout_sec: int = _env_int("QUICK_TIMEOUT_SEC", 8)

    web_cache_ttl_hours: int = _env_int("WEB_CACHE_TTL_HOURS", 12)
    web_cache_max_chars_per_page: int = _env_int("WEB_CACHE_MAX_CHARS", 2600)
    web_live_fallback: bool = os.getenv("WEB_LIVE_FALLBACK", "false").lower() == "true"
    spatial_tool_enabled: bool = _env_bool("SPATIAL_TOOL_ENABLED", True)
    spatial_tool_strict_intent: bool = _env_bool("SPATIAL_TOOL_STRICT_INTENT", True)

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _env_int("PORT", _env_int("APP_PORT", 5000))


CONFIG = AppConfig()


def groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


def pdf_paths() -> List[str]:
    candidates = [
        "CICTify - FAQs.pdf",
        "BulSU Student handbook.pdf",
        "Faculty Manual for BOR.pdf",
        "BulSU-Enhanced-Guidelines.pdf",
        "UnivCalendar_2526.pdf",
        "UniversityCalendar_AY2526.pdf",
        "guide.pdf",
        "CICT Rooms - Pics & Desc.docx.pdf",
        "CICT-Rooms.pdf",
    ]

    paths: List[str] = []
    for name in candidates:
        kb_path = KB_PDF_DIR / name
        root_path = BASE_DIR / name
        if kb_path.exists():
            paths.append(str(kb_path))
        elif root_path.exists():
            paths.append(str(root_path))

    if KB_PDF_DIR.exists():
        for extra in sorted(KB_PDF_DIR.glob("*.pdf")):
            extra_str = str(extra)
            if extra_str not in paths:
                paths.append(extra_str)

    # Avoid duplicated guideline content when both files are present.
    enhanced = str(KB_PDF_DIR / "BulSU-Enhanced-Guidelines.pdf")
    guide = str(KB_PDF_DIR / "guide.pdf")
    if enhanced in paths and guide in paths:
        paths = [p for p in paths if p != guide]

    # Remove empty files from the corpus list.
    paths = [p for p in paths if pathlib.Path(p).exists() and pathlib.Path(p).stat().st_size > 0]

    return paths
