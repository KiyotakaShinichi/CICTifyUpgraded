import asyncio
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .config import CONFIG
from .floorplan_context import FloorplanContextStore
from .llm import GroqClient
from .prompts import (
    rag_system_prompt,
    web_system_prompt,
)
from .retrieval import RAGStore
from .safety import AdvancedSecurityGuard, ResponseFilter, validate_model_response
from .spatial_graph import SpatialGraphStore
from .web_search import WebFallback


ROUTER_PROMPT = """You are a multilingual intent router for BulSU CICT assistant.

Task:
Classify the user query into exactly one label:
- GENERAL: greetings, identity, small talk, thanks, short chit-chat.
- ACADEMIC: any BulSU/CICT information request, policies, procedures, announcements, schedules, admissions, grading, room/location, requirements, offices, faculty, or factual queries.

Rules:
- Support multilingual input (English, Filipino/Tagalog, Taglish, and similar variations).
- If the query is not clearly casual, choose ACADEMIC.
- Output exactly one token only: GENERAL or ACADEMIC.

Examples:
- "hi" -> GENERAL
- "huy" -> GENERAL
- "who are you" -> GENERAL
- "ano name mo" -> GENERAL
- "salamat" -> GENERAL
- "good morning" -> GENERAL
- "ano requirements sa enrollment" -> ACADEMIC
- "where is the dean's office" -> ACADEMIC
- "paano mag shift to BSIT" -> ACADEMIC
- "sino dean ng cict" -> ACADEMIC
- "what is the grading policy" -> ACADEMIC
"""

FALLBACK_GENERAL_PATTERNS = [
    r"^(hi|hello|hey|heya|hii+|helo+|huy|hu|yo|sup|kumusta|kamusta)\b",
    r"^good\s+(morning|afternoon|evening)\b",
    r"^(who\s+are\s+you|what'?s\s+your\s+name|sino\s+ka|ano\s+(pangalan|name)\s+mo)\b",
    r"^(where\s+are\s+you|nasaan\s+ka|nasan\s+ka)\b",
    r"^(what\s+languages?\s+do\s+you\s+speak|anong\s+wika|language\s+mo)\b",
    r"^(thanks|thank\s+you|salamat|ok|okay|sige|bye|goodbye|ingat)\b",
]

ACADEMIC_OVERRIDE_TERMS = {
    "bulsu", "cict", "dean", "coordinator", "adviser", "advisor", "faculty", "registrar", "cashier",
    "enroll", "enrollment", "requirement", "requirements", "admission", "transferee", "shiftee", "returnee",
    "bsit", "bscs", "room", "laboratory", "lab", "acad", "schedule", "policy", "grading", "tuition",
    "office", "announcement", "curriculum", "subject", "courses",
}


@dataclass
class CascadeResult:
    reply: str
    route: str
    context: str


class CascadeOrchestrator:
    def __init__(self) -> None:
        self.llm = GroqClient()
        self.rag = RAGStore()
        self.floorplans = FloorplanContextStore()
        self.spatial_graph = SpatialGraphStore()
        self.web = WebFallback()

    def initialize(self) -> None:
        prewarm_timeout = max(20, CONFIG.request_timeout_sec * 3)

        async def _prewarm() -> None:
            await asyncio.wait_for(self.web.prewarm(force=False), timeout=prewarm_timeout)

        self.rag.load_or_build()
        try:
            asyncio.run(_prewarm())
            print(f"[Startup] Web prewarm complete (timeout={prewarm_timeout}s)", flush=True)
        except asyncio.TimeoutError:
            print(f"[Startup] Web prewarm timed out after {prewarm_timeout}s; continuing startup", flush=True)
        except RuntimeError:
            # Fallback for environments where an event loop is already active.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_prewarm())
                print(f"[Startup] Web prewarm complete (timeout={prewarm_timeout}s)", flush=True)
            except asyncio.TimeoutError:
                print(f"[Startup] Web prewarm timed out after {prewarm_timeout}s; continuing startup", flush=True)
            finally:
                loop.close()

    async def _route(self, question: str) -> str:
        q = (question or "").strip()
        label = await self.llm.chat(
            ROUTER_PROMPT,
            q,
            model=CONFIG.routing_model,
            max_tokens=8,
            temperature=0.0,
            timeout_sec=CONFIG.quick_timeout_sec,
        )
        label_upper = (label or "").upper()
        q_lower = q.lower()
        if "GENERAL" in label_upper:
            if self._looks_academic(q_lower):
                return "ACADEMIC"
            return "GENERAL"

        if "ACADEMIC" in label_upper:
            return "ACADEMIC"

        # Safety net only if router output is missing/invalid.
        if any(re.match(pattern, q_lower) for pattern in FALLBACK_GENERAL_PATTERNS):
            if self._looks_academic(q_lower):
                return "ACADEMIC"
            return "GENERAL"
        return "ACADEMIC"

    @staticmethod
    def _looks_academic(text: str) -> bool:
        tokens = set(re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower()))
        return any(term in tokens for term in ACADEMIC_OVERRIDE_TERMS)

    @staticmethod
    def _deterministic_general_reply(question: str) -> str:
        q = (question or "").strip().lower()

        if re.search(r"\b(hi|hello|helo|hey|huy|kumusta|kamusta|good\s+morning|good\s+afternoon|good\s+evening)\b", q):
            return "Hello. I am CICTify, your BulSU CICT assistant."

        if re.search(r"\b(who\s+are\s+you|what'?s\s+your\s+name|sino\s+ka|ano\s+(pangalan|name)\s+mo)\b", q):
            return "I am CICTify, a virtual assistant focused on BulSU CICT information."

        if re.search(r"\b(where\s+are\s+you|nasaan\s+ka|nasan\s+ka|asan\s+ka)\b", q):
            return "I do not have a physical location. I am a virtual assistant for BulSU CICT."

        if re.search(r"\b(language|languages|wika|tagalog|filipino|english|taglish)\b", q):
            return "I can assist in English, Filipino/Tagalog, and Taglish."

        if re.search(r"\b(thanks|thank\s+you|salamat|bye|goodbye|ingat)\b", q):
            return "You are welcome."

        return "I can help with BulSU CICT questions."

    @staticmethod
    def _expand_room_tokens(text: str) -> str:
        raw = text or ""
        pairs = re.findall(r"\b([a-zA-Z]+)(\d+)\b", raw)
        if not pairs:
            return raw

        expansions = [f"{letters} {digits}" for letters, digits in pairs]
        return raw + "\n" + " ".join(expansions)

    @staticmethod
    def _history_to_text(history: List[Dict]) -> str:
        recent = history[-CONFIG.max_history_messages:]
        lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _preview_from_rag_context(context: str, max_chars: int = 520) -> str:
        if not context:
            return ""

        first_block = context.split("\n\n---\n\n", 1)[0].strip()
        source = "Unknown"
        content = first_block

        if first_block.startswith("Source:"):
            lines = first_block.splitlines()
            if lines:
                source = lines[0].replace("Source:", "", 1).strip() or "Unknown"
                content = "\n".join(lines[1:]).strip()

        content = re.sub(r"\s+", " ", content).strip()
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."

        if not content:
            return ""
        return f"{content}"

    @staticmethod
    def _contains_followup_marker(question: str) -> bool:
        q = (question or "").strip().lower()
        markers = [
            "so", "only", "that", "those", "it", "they", "them", "what about", "her", "his", "she", "he",
            "yun", "iyon", "ganon", "ganoon", "so ibig", "ibig sabihin", "pwede pa", "pano naman",
        ]
        return any(m in q for m in markers)

    @staticmethod
    def _direct_fact_fallback(question: str, rag_context: str) -> str:
        q = (question or "").lower()
        flat = re.sub(r"\s+", " ", rag_context or "")
        normalized_flat = unicodedata.normalize("NFKD", flat)
        if not flat:
            return ""

        if "transferee" in q:
            m = re.search(r"Transferee\s+is\s+([^\.]{20,260})\.", flat, flags=re.IGNORECASE)
            if m:
                return (
                    f"Transferee: {m.group(1).strip()}.\n\n"
                    "Source: BulSU-Enhanced-Guidelines.pdf"
                )

        if "shiftee" in q:
            m = re.search(r"Shiftee\s+is\s+([^\.]{20,260})\.", flat, flags=re.IGNORECASE)
            if m:
                return (
                    f"Shiftee: {m.group(1).strip()}.\n\n"
                    "Source: BulSU-Enhanced-Guidelines.pdf"
                )

        if "returnee" in q:
            m = re.search(r"Returnee\s+is\s+([^\.]{20,280})\.", flat, flags=re.IGNORECASE)
            if m:
                return (
                    f"Returnee: {m.group(1).strip()}.\n\n"
                    "Source: BulSU-Enhanced-Guidelines.pdf"
                )

        dean_name_query = ("dean" in q and ("name" in q or "who" in q)) or ("her name" in q) or ("his name" in q)
        if dean_name_query:
            m = re.search(r"(Dr\.\s+[A-Z][A-Za-z\.\s]{2,60}?)\s*(?:-|,)?\s*Dean\b", flat)
            if not m:
                # OCR may drop punctuation around role labels.
                m = re.search(r"Dean\s*,?\s*CICT\s*[\.|,\-]?\s*(Dr\.\s+[A-Z][A-Za-z\.\s]{2,60})", flat)
            if m:
                name = (m.group(1) or "").strip()
                name = re.sub(r"\s+", " ", name)
                return f"CICT Dean: {name}.\n\nSource: CICTify - FAQs.pdf"

        if "coordinator" in q:
            matches = re.findall(
                r"((?:Dr|Mr|Ms|Engr)\s*\.\s*[A-Z][A-Za-z0-9\.\s]{2,80})[^\.\n]{0,100}?Program\s*Coordinator\s*,?\s*([A-Za-z&\-\s]{3,90})",
                normalized_flat,
                flags=re.IGNORECASE,
            )
            if matches:
                seen = set()
                lines = []
                for name, track in matches:
                    clean_name = re.sub(r"\s+", " ", name).strip()
                    clean_track = re.sub(r"\s+", " ", track).strip(" .,")
                    key = (clean_name.lower(), clean_track.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"- {clean_name}: {clean_track}")
                if lines:
                    return "CICT BSIT Program Coordinators:\n" + "\n".join(lines[:5]) + "\n\nSource: CICTify - FAQs.pdf"

        if ("acad" in q or "room" in q) and "where" in q:
            m = re.search(r"(Acad\s*1\s*[\-–]\s*[^\.]{20,220})", flat, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip() + "."

        return ""

    @staticmethod
    def _extractive_rag_fallback(question: str, rag_context: str, max_sentences: int = 3) -> str:
        if not rag_context:
            return ""

        expanded_q = CascadeOrchestrator._expand_room_tokens(question)
        q_terms = set(re.findall(r"[a-zA-Z0-9]{2,}", expanded_q.lower()))
        if not q_terms:
            return ""

        blocks = rag_context.split("\n\n---\n\n")
        scored = []
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            source = "Unknown"
            body = " ".join(lines)
            if lines and lines[0].startswith("Source:"):
                source = lines[0].replace("Source:", "", 1).strip() or "Unknown"
                body = " ".join(lines[1:])

            sentences = re.split(r"(?<=[.!?])\s+", body)
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 20:
                    continue
                s_lower = sent_clean.lower()
                score = sum(1 for term in q_terms if term in s_lower)
                if score > 0:
                    scored.append((score, source, sent_clean))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        picked = scored[:max_sentences]
        lines = [f"- {sentence}" for _, _, sentence in picked]
        primary_source = picked[0][1]
        return "\n".join(lines) + f"\n\nSource: {primary_source}"

    @staticmethod
    def _keywords(text: str) -> List[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "have", "what", "where", "when",
            "which", "who", "why", "how", "can", "you", "your", "are", "is", "to", "of", "in",
            "ang", "mga", "ano", "saan", "sino", "paano", "pwede", "ba", "po", "ng", "sa", "at",
        }
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())
        return [t for t in tokens if t not in stop]

    def _relevant_memory(self, question: str, history: List[Dict]) -> str:
        if not history:
            return ""

        q_terms = set(self._keywords(question))
        followup = self._contains_followup_marker(question)
        candidates: List[Tuple[int, int, str]] = []
        recent = history[:-1][-CONFIG.max_history_messages:]

        for idx, msg in enumerate(recent):
            if msg.get("role") != "user":
                continue
            user_text = (msg.get("content") or "").strip()
            if not user_text:
                continue

            user_terms = set(self._keywords(user_text))
            overlap = len(q_terms.intersection(user_terms)) if q_terms else 0
            recency_boost = idx + 1
            score = overlap * 3 + recency_boost
            if followup:
                score += 2

            if overlap >= CONFIG.memory_min_score or followup:
                assistant_text = ""
                if idx + 1 < len(recent) and recent[idx + 1].get("role") == "assistant":
                    assistant_text = (recent[idx + 1].get("content") or "").strip()
                block = f"User: {user_text}"
                if assistant_text:
                    block += f"\nAssistant: {assistant_text}"
                candidates.append((score, idx, block))

        if not candidates:
            return ""

        picked = sorted(candidates, key=lambda item: item[0], reverse=True)[: CONFIG.memory_top_k]
        picked.sort(key=lambda item: item[1])
        return "\n\n".join(block for _, _, block in picked)

    async def respond(self, question: str, history: List[Dict]) -> Tuple[CascadeResult, str]:
        print(f"[Query] {question}", flush=True)
        safety = await AdvancedSecurityGuard.validate_message(question, self.llm)
        if safety.get("block"):
            return CascadeResult(
                reply=AdvancedSecurityGuard.get_security_response(safety.get("category", "unknown")),
                route="SAFETY_BLOCK",
                context=safety.get("reason", "blocked"),
            ), safety.get("reason", "blocked")

        route = await self._route(question)
        print(f"[Router] -> {route}", flush=True)
        history_text = self._history_to_text(history)
        relevant_memory = self._relevant_memory(question, history)
        if relevant_memory:
            print("[Memory] Relevant history found for current query", flush=True)

        if route == "GENERAL":
            final = self._deterministic_general_reply(question)
            final = ResponseFilter.filter_response(final)
            validate = validate_model_response(final)
            if validate.blocked:
                final = "I can only provide safe BulSU CICT assistance."
            return CascadeResult(reply=final, route="GENERAL", context=""), "ok"

        spatial_reply = self.spatial_graph.answer_navigation_query(question)
        if spatial_reply:
            return CascadeResult(reply=spatial_reply, route="SPATIAL_RAG", context="spatial_graph"), "ok"

        # Cascading workflow: GENERAL -> RAG -> WEB
        print("[Cascade] Trying RAG", flush=True)
        retrieval_query = self._expand_room_tokens(question)
        if relevant_memory:
            retrieval_query = f"{self._expand_room_tokens(question)}\n\nRelevant previous conversation:\n{relevant_memory}"

        rag_context = self.rag.context_for(retrieval_query)
        rag_chunks = rag_context.count("Source:") if rag_context else 0
        print(f"[Cascade] RAG chunks={rag_chunks}", flush=True)
        rag_preview = ""
        if rag_context:
            rag_user_prompt = f"Current Question: {question}"
            if relevant_memory:
                rag_user_prompt = (
                    "Context from previous conversation (only use if necessary to understand the current question):\n"
                    f"{relevant_memory}\n\n"
                    f"Current Question: {question}"
                )
            rag_answer = await self.llm.chat_with_fallbacks(
                rag_system_prompt(rag_context),
                rag_user_prompt,
            )
            if rag_answer and "__NO_KB_ANSWER__" not in rag_answer:
                rag_answer = ResponseFilter.filter_response(rag_answer)
                validate = validate_model_response(rag_answer)
                if not validate.blocked:
                    return CascadeResult(reply=rag_answer, route="RAG", context=rag_context), "ok"

            rag_preview = self._preview_from_rag_context(rag_context)
            if not rag_answer:
                print("[Cascade] RAG answer empty; trying WEB before fallback preview", flush=True)

        print("[Cascade] Trying WEB", flush=True)
        web_context = await self.web.search(question)
        web_hits = web_context.count("From https://") if web_context else 0
        print(f"[Cascade] WEB hits={web_hits}", flush=True)
        if web_context:
            web_user_prompt = f"Question: {question}"
            if relevant_memory:
                web_user_prompt = (
                    "Relevant conversation memory:\n"
                    f"{relevant_memory}\n\n"
                    f"Question: {question}"
                )
            web_answer = await self.llm.chat_with_fallbacks(
                web_system_prompt(web_context),
                web_user_prompt,
            )
            if web_answer and "__NO_WEB_ANSWER__" not in web_answer:
                web_answer = ResponseFilter.filter_response(web_answer)
                validate = validate_model_response(web_answer)
                if not validate.blocked:
                    return CascadeResult(reply=web_answer, route="WEB", context=web_context), "ok"

        if rag_preview:
            return CascadeResult(
                reply=(
                    "I found some related information in the knowledge base, but it might not answer your question fully:\n\n"
                    f"> {rag_preview}"
                ),
                route="RAG_PREVIEW",
                context=rag_context,
            ), "ok"

        return CascadeResult(
            reply="Wala sa knowledge ko ngayon ang sagot sa tanong na iyan based on my BulSU CICT knowledge base and prewarmed web sources.",
            route="NO_RESULT",
            context="",
        ), "ok"

    async def close(self) -> None:
        await self.llm.close()
