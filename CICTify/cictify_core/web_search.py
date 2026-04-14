import asyncio
import json
import pathlib
import re
from datetime import datetime, timedelta
from typing import List

import aiohttp
from bs4 import BeautifulSoup

from .config import BASE_DIR, CONFIG


class WebFallback:
    BASE_URLS = [
        "https://bulsucict.com/",
        "https://bulsucict.com/about-us/",
        "https://bulsucict.com/announcement/",
        "https://bulsucict.com/cict-faculty/",
        "https://bulsucict.com/news-and-updates/",
    ]

    def __init__(self) -> None:
        self._cache_path = BASE_DIR / "vectorstore" / "web_cache.json"
        self._cached_pages: List[dict] = []

    def _load_cache_from_disk(self) -> None:
        if not self._cache_path.exists():
            self._cached_pages = []
            return
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            pages = raw.get("pages", []) if isinstance(raw, dict) else []
            self._cached_pages = [p for p in pages if p.get("url") and p.get("text")]
        except Exception:
            self._cached_pages = []

    def _cache_is_fresh(self) -> bool:
        if not self._cache_path.exists():
            return False
        modified = datetime.fromtimestamp(self._cache_path.stat().st_mtime)
        return datetime.now() - modified < timedelta(hours=CONFIG.web_cache_ttl_hours)

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(),
            "pages": self._cached_pages,
        }
        self._cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    async def _fetch(self, session: aiohttp.ClientSession, url: str) -> str:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status != 200:
                    return ""
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                    tag.decompose()
                text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
                return text[:CONFIG.web_cache_max_chars_per_page]
        except asyncio.TimeoutError:
            return ""
        except Exception:
            return ""

    async def prewarm(self, force: bool = False) -> None:
        if not force and self._cache_is_fresh():
            self._load_cache_from_disk()
            if self._cached_pages:
                return

        pages: List[dict] = []
        async with aiohttp.ClientSession() as session:
            for url in self.BASE_URLS:
                text = await self._fetch(session, url)
                if text:
                    pages.append({"url": url, "text": text})

        if pages:
            self._cached_pages = pages
            self._save_cache()
        else:
            self._load_cache_from_disk()

    @staticmethod
    def _normalize_terms(question: str) -> List[str]:
        terms = re.findall(r"[a-zA-Z0-9]{3,}", (question or "").lower())
        return [t for t in terms if t not in {"what", "where", "when", "which", "that", "with", "from", "have"}]

    def _top_relevant_pages(self, question: str, top_k: int = 2) -> List[dict]:
        terms = self._normalize_terms(question)
        if not terms:
            return self._cached_pages[:top_k]

        scored: List[tuple[int, dict]] = []
        for page in self._cached_pages:
            text = page.get("text", "").lower()
            score = sum(text.count(term) for term in terms)
            scored.append((score, page))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [page for score, page in scored if score > 0][:top_k]
        return selected or [page for _, page in scored[:top_k]]

    async def search(self, question: str) -> str:
        if not self._cached_pages:
            self._load_cache_from_disk()

        if not self._cached_pages and CONFIG.web_live_fallback:
            await self.prewarm(force=True)

        if not self._cached_pages:
            return ""

        pages = self._top_relevant_pages(question)
        snippets = [f"From {page['url']}: {page['text']}" for page in pages]
        return "\n\n".join(snippets)
