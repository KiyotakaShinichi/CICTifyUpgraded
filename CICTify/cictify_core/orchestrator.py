import asyncio
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .config import CONFIG
from .floorplan_context import FloorplanContextStore
from .llm import GroqClient
from .prompts import (
    general_system_prompt,
    rag_system_prompt,
    web_system_prompt,
)
from .retrieval import RAGStore
from .safety import AdvancedSecurityGuard, ResponseFilter, validate_model_response
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
    r"^(thanks|thank\s+you|salamat|ok|okay|sige|bye|goodbye|ingat)\b",
]


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
        self.web = WebFallback()

    def initialize(self) -> None:
        self.rag.load_or_build()
        try:
            asyncio.run(self.web.prewarm(force=False))
        except RuntimeError:
            # Fallback for environments where an event loop is already active.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.web.prewarm(force=False))
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
        if "GENERAL" in label_upper:
            return "GENERAL"

        if "ACADEMIC" in label_upper:
            return "ACADEMIC"

        # Safety net only if router output is missing/invalid.
        q_lower = q.lower()
        if any(re.match(pattern, q_lower) for pattern in FALLBACK_GENERAL_PATTERNS):
            return "GENERAL"
        return "ACADEMIC"

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
        return f"Source: {source}\n{content}"

    @staticmethod
    def _contains_followup_marker(question: str) -> bool:
        q = (question or "").strip().lower()
        markers = [
            "so", "only", "that", "those", "it", "they", "them", "what about",
            "yun", "iyon", "ganon", "ganoon", "so ibig", "ibig sabihin", "pwede pa", "pano naman",
        ]
        return any(m in q for m in markers)

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
        followup = self._contains_followup_marker(question) or len((question or "").split()) <= 5
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
            prompt = (
                "Conversation history:\n"
                f"{history_text}\n\n"
                "Relevant memory:\n"
                f"{relevant_memory or 'None'}\n\n"
                "User question:\n"
                f"{question}"
            )
            answer = await self.llm.chat_with_fallbacks(general_system_prompt(), prompt, max_tokens=700)
            final = answer or "Hello. I am CICTify. How can I help with BulSU CICT today?"
            final = ResponseFilter.filter_response(final)
            validate = validate_model_response(final)
            if validate.blocked:
                final = "I can only provide safe BulSU CICT assistance."
            return CascadeResult(reply=final, route="GENERAL", context=""), "ok"

        # Cascading workflow: GENERAL -> RAG -> WEB
        print("[Cascade] Trying RAG", flush=True)
        retrieval_query = question
        if relevant_memory:
            retrieval_query = f"{question}\n\nRelevant previous conversation:\n{relevant_memory}"

        rag_context = self.rag.context_for(retrieval_query)
        rag_chunks = rag_context.count("Source:") if rag_context else 0
        print(f"[Cascade] RAG chunks={rag_chunks}", flush=True)
        rag_preview = ""
        if rag_context:
            rag_user_prompt = f"Question: {question}"
            if relevant_memory:
                rag_user_prompt = (
                    "Relevant conversation memory:\n"
                    f"{relevant_memory}\n\n"
                    f"Question: {question}"
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
                    "May nakita akong related context sa knowledge base pero hindi ako nakabuo ng final answer ngayon. "
                    "Try ulit in a moment.\n\n"
                    f"{rag_preview}"
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
