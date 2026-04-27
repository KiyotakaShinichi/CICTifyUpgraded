import asyncio
import json
import random
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
from .spatial_tool import SpatialAwarenessTool
from .web_search import WebFallback


ROUTER_PROMPT = """You are a multilingual intent router for BulSU CICT assistant.

Task:
Classify the user query into exactly one intent label:
- GENERAL: greetings, identity, small talk, thanks, short chit-chat.
- ACADEMIC: factual/policy/procedure/location/people/schedule/admission/academic questions.

Primary principle:
- Use semantic understanding first. Heuristic hints are secondary.

Examples:
- "hi", "kumusta", "thanks" -> GENERAL
- "who is the dean", "what is a transferee", "where is registrar" -> ACADEMIC
- "u sure?" after an answer -> GENERAL (confirmation follow-up)
- "how about vice dean" after role query -> ACADEMIC (contextual follow-up)

Return strict JSON only:
{
    "label": "GENERAL" | "ACADEMIC",
    "confidence": 0.0-1.0,
    "is_followup": true|false,
    "reason": "short reason"
}
"""

SCOPE_PROMPT = """You are a scope classifier for BulSU CICT assistant.

Task:
Decide if the query is within BulSU CICT scope.

Rules:
- If query is BulSU/CICT related, greetings, or a clear follow-up to prior BulSU context -> IN_SCOPE.
- If query is unrelated general-knowledge (e.g., world history, random math, coding help outside BulSU context) -> OUT_OF_SCOPE.
- Use semantics first; keyword hints are only supportive.

Return strict JSON only:
{
    "scope": "IN_SCOPE" | "OUT_OF_SCOPE",
    "confidence": 0.0-1.0,
    "reason": "short reason"
}
"""

FALLBACK_GENERAL_PATTERNS = [
    r"^(hi|hello|hey|heya|hii+|helo+|huy|hu|yo|sup|kumusta|kamusta|bonjour|hola|salut|ciao|guten\s+tag)\b",
    r"^good\s+(morning|afternoon|evening)\b",
    r"^(how\s+are\s+you|how\s+r\s+u|kamusta\s+ka|kumusta\s+ka|hvordan\s+gar\s+det)\b",
    r"^(who\s+are\s+you|what'?s\s+your\s+name|sino\s+ka|ano\s+(pangalan|name)\s+mo|qui\s+es-tu|wer\s+bist\s+du|chi\s+sei)\b",
    r"^(where\s+are\s+you|nasaan\s+ka|nasan\s+ka|asan\s+ka|ou\s+es-tu|wo\s+bist\s+du|dove\s+sei)\b",
    r"^(what\s+languages?\s+do\s+you\s+speak|anong\s+wika|language\s+mo|quelles\s+langues|welche\s+sprachen|quali\s+lingue)\b",
    r"^(thanks|thank\s+you|salamat|ok|okay|sige|bye|goodbye|ingat|merci|danke|grazie)\b",
    r"^(ah|oh|okay|ok|sige|noted|gets)?[\s,!.]*(thanks|thank\s+you|salamat|merci|danke|grazie)\b",
]

ACADEMIC_OVERRIDE_TERMS = {
    "bulsu", "cict", "dean", "coordinator", "adviser", "advisor", "faculty", "registrar", "cashier",
    "enroll", "enrollment", "requirement", "requirements", "admission", "transferee", "shiftee", "returnee",
    "bsit", "bscs", "room", "laboratory", "lab", "acad", "schedule", "policy", "grading", "tuition",
    "office", "announcement", "curriculum", "subject", "courses", "tenure", "promotion", "visitor",
    "faculty", "staff", "employee", "hrmo", "guidelines", "academic", "research", "extension",
    "pimentel", "nstp", "building", "hall", "hostel", "dorm", "location", "directions", "route", "near",
}

OUT_OF_SCOPE_PATTERNS = [
    r"\bwho\s+is\s+hitler\b",
    r"\bwho\s+is\s+napoleon\b",
    r"\bwhat\s+is\s+\d+\s*[\+\-\*/]\s*\d+\b",
    r"\bwwhat\s+is\s+\d+\s*[\+\-\*/]\s*\d+\b",
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
        self.spatial_graph = SpatialGraphStore()
        self.spatial_tool = SpatialAwarenessTool(self.spatial_graph)
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

    @staticmethod
    def _parse_json_object(raw: str) -> Dict:
        text = str(raw or "").strip()
        if not text:
            return {}
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            text = text[start : end + 1]
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    async def _route(self, question: str) -> str:
        q = (question or "").strip()
        q_lower = q.lower()

        general_hint = bool(any(re.match(pattern, q_lower) for pattern in FALLBACK_GENERAL_PATTERNS))
        academic_hint = bool(self._looks_academic(q_lower))
        followup_hint = bool(self._contains_followup_marker(q_lower) or self._is_confirmation_followup(q_lower))

        router_user_prompt = (
            f"Query: {q}\n"
            f"Heuristic hints (secondary only): general_hint={general_hint}, academic_hint={academic_hint}, followup_hint={followup_hint}"
        )

        raw = await self.llm.chat(
            ROUTER_PROMPT,
            router_user_prompt,
            model=CONFIG.routing_model,
            max_tokens=120,
            temperature=0.0,
            timeout_sec=CONFIG.quick_timeout_sec,
        )

        parsed = self._parse_json_object(raw or "")
        llm_label = str(parsed.get("label") or "").strip().upper()
        try:
            llm_conf = float(parsed.get("confidence", 0.0) or 0.0)
        except Exception:
            llm_conf = 0.0

        heuristic_label = "ACADEMIC"
        if general_hint and not academic_hint:
            heuristic_label = "GENERAL"
        elif academic_hint and not general_hint:
            heuristic_label = "ACADEMIC"
        elif followup_hint and not academic_hint:
            heuristic_label = "GENERAL"

        if llm_label in {"GENERAL", "ACADEMIC"}:
            # LLM is primary. Heuristics only assist when confidence is low and conflict is strong.
            if llm_conf < 0.55 and llm_label != heuristic_label:
                return heuristic_label
            return llm_label

        # Malformed router output fallback.
        return heuristic_label or "ACADEMIC"

    async def _is_out_of_scope_hybrid(self, question: str, relevant_memory: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False

        if relevant_memory and self._is_known_place_phrase(q):
            return False

        # Deterministic high-signal cases remain hard blocks.
        if any(re.search(pattern, q) for pattern in OUT_OF_SCOPE_PATTERNS):
            return True

        followup_hint = bool(self._contains_followup_marker(q) or self._is_confirmation_followup(q))
        academic_hint = bool(self._looks_academic(q))
        direct_wh = bool(re.search(r"\b(who\s+is|what\s+is|where\s+is|when\s+is|why\s+is|how\s+to)\b", q))

        scope_user_prompt = (
            f"Query: {question}\n"
            f"Recent memory available: {bool(relevant_memory)}\n"
            f"Heuristic hints (secondary only): followup_hint={followup_hint}, academic_hint={academic_hint}, direct_wh={direct_wh}"
        )

        raw = await self.llm.chat(
            SCOPE_PROMPT,
            scope_user_prompt,
            model=CONFIG.routing_model,
            max_tokens=120,
            temperature=0.0,
            timeout_sec=CONFIG.quick_timeout_sec,
        )
        parsed = self._parse_json_object(raw or "")
        scope = str(parsed.get("scope") or "").strip().upper()
        try:
            conf = float(parsed.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0

        heuristic_out = False
        if followup_hint and relevant_memory:
            heuristic_out = False
        elif any(re.match(pattern, q) for pattern in FALLBACK_GENERAL_PATTERNS):
            heuristic_out = False
        elif academic_hint:
            heuristic_out = False
        elif re.search(r"\b(where\s+is|nasaan|asan|nasan|near|nearby|beside|route|path|directions|papunta|paano\s+pumunta)\b", q):
            heuristic_out = False
        elif direct_wh:
            heuristic_out = True

        if scope in {"IN_SCOPE", "OUT_OF_SCOPE"}:
            llm_out = scope == "OUT_OF_SCOPE"
            if conf < 0.55:
                return heuristic_out
            return llm_out

        return heuristic_out

    async def _is_context_relevant(self, question: str, rag_context: str) -> bool:
        context = (rag_context or "").strip()
        if not context:
            return False
        judge_prompt = (
            "You are a strict relevance judge for BulSU CICT QA. "
            "Answer with exactly one token: RELEVANT or NOT_RELEVANT.\n"
            "Mark RELEVANT only if the provided context directly answers the user's question "
            "or gives clearly supporting facts. If context is only loosely related, noisy, or unrelated, "
            "return NOT_RELEVANT."
        )
        user_prompt = f"Question:\n{question}\n\nContext:\n{context[:3500]}"
        verdict = await self.llm.chat(
            judge_prompt,
            user_prompt,
            model=CONFIG.routing_model,
            max_tokens=6,
            temperature=0.0,
            timeout_sec=CONFIG.quick_timeout_sec,
        )
        return "RELEVANT" in (verdict or "").upper() and "NOT_RELEVANT" not in (verdict or "").upper()

    @staticmethod
    def _looks_academic(text: str) -> bool:
        tokens = set(re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower()))
        return any(term in tokens for term in ACADEMIC_OVERRIDE_TERMS)

    @staticmethod
    def _deterministic_general_reply(question: str) -> str:
        q = (question or "").strip().lower()
        if re.search(r"\b(bonjour|salut)\b", q):
            return "Bonjour. Je suis CICTify, votre assistant BulSU CICT."
        if re.search(r"\b(guten\s+tag|hallo)\b", q):
            return "Hallo. Ich bin CICTify, dein BulSU CICT Assistent."
        if re.search(r"\b(hola|buenos\s+dias)\b", q):
            return "Hola. Soy CICTify, tu asistente de BulSU CICT."
        if re.search(r"\b(how\s+are\s+you|how\s+r\s+u|kamusta\s+ka|kumusta\s+ka|hvordan\s+gar\s+det)\b", q):
            return "I am doing well and ready to help with BulSU CICT questions."
        greeting_options = [
            "Hello. I am CICTify, your BulSU CICT assistant.",
            "Hi there. I am CICTify, ready to help with BulSU CICT concerns.",
            "Greetings. I am CICTify, your BulSU CICT virtual assistant.",
            "Hello! CICTify here. I can help with BulSU CICT questions.",
        ]

        if re.search(r"\b(hi|hello|helo|hey|huy|kumusta|kamusta|good\s+morning|good\s+afternoon|good\s+evening|bonjour|hola|salut|ciao|guten\s+tag)\b", q):
            return random.choice(greeting_options)

        if re.search(r"\b(who\s+are\s+you|what'?s\s+your\s+name|sino\s+ka|ano\s+(pangalan|name)\s+mo)\b", q):
            return "I am CICTify, a virtual assistant focused on BulSU CICT information."

        if re.search(r"\b(where\s+are\s+you|nasaan\s+ka|nasan\s+ka|asan\s+ka)\b", q):
            return "I do not have a physical location. I am a virtual assistant for BulSU CICT."

        if re.search(r"\b(language|languages|wika|tagalog|filipino|english|taglish|french|spanish|german|italian)\b", q):
            return "I can assist in English, Filipino/Tagalog, Taglish, and basic multilingual greetings."

        if re.search(r"\b(thanks|thank\s+you|salamat|bye|goodbye|ingat|merci|danke|grazie)\b", q):
            return "You are welcome."

        return "I can help with BulSU CICT questions."

    @staticmethod
    def _normalize_question(question: str) -> str:
        q = unicodedata.normalize("NFKC", (question or "")).strip()
        replacements = {
            r"\bashiftee\b": "a shiftee",
            r"\bshifteee\b": "shiftee",
            r"\btransferee\b": "transferee",
            r"\bdeen\b": "dean",
            r"\bwwhat\b": "what",
            r"\bhow\s+a\s*bout\b": "how about",
            r"\ba\s+bout\b": "about",
            r"\bu\b": "you",
            r"\bur\b": "your",
            r"\bpls\b": "please",
            r"\bpls\.?\b": "please",
            r"\breqs\b": "requirements",
            r"\benrol\b": "enroll",
            r"\benrollmnt\b": "enrollment",
        }
        for pattern, repl in replacements.items():
            q = re.sub(pattern, repl, q, flags=re.IGNORECASE)
        q = re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    @staticmethod
    def _coalesce_queries(*items: str) -> str:
        seen = set()
        parts: List[str] = []
        for item in items:
            chunk = (item or "").strip()
            if not chunk:
                continue
            key = chunk.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(chunk)
        return "\n".join(parts)

    def _is_known_place_phrase(self, text: str) -> bool:
        phrase = str(text or "").strip()
        if not phrase or len(phrase) > 80:
            return False
        try:
            return bool(
                self.spatial_graph._match_node_id(phrase)
                or self.spatial_graph._match_building_name(phrase)
            )
        except Exception:
            return False

    @staticmethod
    def _last_user_query_from_memory(relevant_memory: str) -> str:
        user_lines = [ln for ln in str(relevant_memory or "").splitlines() if ln.startswith("User:")]
        if not user_lines:
            return ""
        return user_lines[-1].replace("User:", "", 1).strip()

    @staticmethod
    def _destination_hint_from_query(text: str) -> str:
        q = str(text or "").strip()
        if not q:
            return ""
        m = re.search(r"(?:where\s+is|nasaan\s+ang|nasaan|asan|nasan)\s+(.+)$", q, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" ?!.,")
        m2 = re.search(r"(?:to|papunta\s+sa)\s+([a-zA-Z0-9/\-\s']+)$", q, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).strip(" ?!.,")
        return ""

    def _resolve_location_only_followup(self, question: str, relevant_memory: str) -> str:
        q = str(question or "").strip()
        if not q or not self._is_known_place_phrase(q) or not relevant_memory:
            return q

        assistant_memory = str(relevant_memory or "").lower()
        asks_location = any(
            marker in assistant_memory
            for marker in (
                "share your current location",
                "current location",
                "step-by-step directions",
                "give directions",
            )
        )
        if not asks_location:
            return q

        last_user = self._last_user_query_from_memory(relevant_memory)
        destination_hint = self._destination_hint_from_query(last_user)
        if not destination_hint:
            return q

        return f"How do I get from {q} to {destination_hint}?"

    @staticmethod
    def _latest_exchange_from_history(history: List[Dict]) -> str:
        if not history:
            return ""
        # Ignore current trailing user turn when reconstructing prior exchange.
        window = history[:-1] if history and history[-1].get("role") == "user" else history[:]
        last_assistant = ""
        for msg in reversed(window):
            if msg.get("role") == "assistant":
                last_assistant = str(msg.get("content") or "").strip()
                break
        if not last_assistant:
            return ""

        last_user = ""
        for msg in reversed(window):
            if msg.get("role") == "user":
                last_user = str(msg.get("content") or "").strip()
                if last_user:
                    break
        if not last_user:
            return f"Assistant: {last_assistant}"
        return f"User: {last_user}\nAssistant: {last_assistant}"

    @staticmethod
    def _merge_contexts(contexts: List[str], max_blocks: int = 10) -> str:
        merged_blocks: List[str] = []
        seen = set()
        for ctx in contexts:
            for block in (ctx or "").split("\n\n---\n\n"):
                b = block.strip()
                if not b:
                    continue
                key = re.sub(r"\s+", " ", b.lower())[:700]
                if key in seen:
                    continue
                seen.add(key)
                merged_blocks.append(b)
                if len(merged_blocks) >= max_blocks:
                    return "\n\n---\n\n".join(merged_blocks)
        return "\n\n---\n\n".join(merged_blocks)

    async def _prepare_query_variants(self, question: str) -> Dict[str, str]:
        q = (question or "").strip()
        if not q:
            return {"corrected_query": "", "english_query": "", "slang_expansion": ""}

        prompt = (
            "Normalize this user query for robust retrieval over BulSU CICT KB and approved web pages. "
            "Fix typos, expand slang/abbreviations, and produce an English variant while preserving original meaning. "
            "Return strict JSON only with keys: corrected_query, english_query, slang_expansion."
        )
        raw = await self.llm.chat(
            "You are a strict JSON generator.",
            f"{prompt}\n\nQuery:\n{q}",
            model=CONFIG.routing_model,
            max_tokens=200,
            temperature=0.0,
            timeout_sec=CONFIG.quick_timeout_sec,
        )

        payload: Dict[str, str] = {
            "corrected_query": q,
            "english_query": q,
            "slang_expansion": q,
        }
        if not raw:
            return payload

        candidate = (raw or "").strip()
        if "{" in candidate and "}" in candidate:
            start = candidate.find("{")
            end = candidate.rfind("}")
            candidate = candidate[start : end + 1]

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                for key in ("corrected_query", "english_query", "slang_expansion"):
                    val = str(parsed.get(key, "") or "").strip()
                    if val:
                        payload[key] = val
        except Exception:
            pass

        return payload

    @staticmethod
    def _detect_language_hint(text: str) -> str:
        t = (text or "").lower()
        if re.search(r"\b(bonjour|salut|merci)\b", t):
            return "fr"
        if re.search(r"\b(oder|nicht|danke|guten)\b", t):
            return "de"
        if re.search(r"\b(kumusta|salamat|sino|ano|paano|pwede|wala)\b", t):
            return "tl"
        return "en"

    @staticmethod
    def _out_of_scope_reply(question: str) -> str:
        lang = CascadeOrchestrator._detect_language_hint(question)
        if lang == "fr":
            return "Je peux seulement aider avec les questions liees a BulSU CICT et les salutations."
        if lang == "de":
            return "Ich kann nur bei BulSU CICT Fragen und Begruessungen helfen."
        if lang == "tl":
            return "BulSU CICT na tanong at greetings lang ang kaya kong sagutin."
        return "I can only help with BulSU CICT questions and greetings."

    @staticmethod
    def _is_confirmation_followup(question: str) -> bool:
        q = (question or "").strip().lower()
        return bool(
            re.search(r"\b(are\s+you\s+sure|u\s+sure|sure\s+tho|sigurado\s+ka|sure\?|bist\s+du\s+sicher)\b", q)
        )

    @staticmethod
    def _latest_assistant_from_history(history: List[Dict]) -> str:
        for msg in reversed(history or []):
            if msg.get("role") == "assistant":
                content = (msg.get("content") or "").strip()
                if content:
                    return content
        return ""

    @staticmethod
    def _confirmation_without_memory(question: str) -> str:
        lang = CascadeOrchestrator._detect_language_hint(question)
        if lang == "fr":
            return "Je peux confirmer si vous precisez a quelle reponse precedente vous faites reference."
        if lang == "de":
            return "Ich kann es bestaetigen, wenn du sagst, auf welche vorherige Antwort du dich beziehst."
        if lang == "tl":
            return "Mako-confirm ko iyan kung sasabihin mo kung aling naunang sagot ang tinutukoy mo."
        return "I can confirm that if you tell me which previous answer you mean."

    @staticmethod
    def _confirmation_from_memory(relevant_memory: str, question: str) -> str:
        if not relevant_memory:
            return ""
        assistant_lines = [ln for ln in relevant_memory.splitlines() if ln.startswith("Assistant:")]
        if not assistant_lines:
            return ""
        last = assistant_lines[-1].replace("Assistant:", "", 1).strip()
        if not last:
            return ""
        lang = CascadeOrchestrator._detect_language_hint(question)
        if lang == "fr":
            return f"Oui. D'apres la reponse precedente: {last}"
        if lang == "de":
            return f"Ja. Basierend auf der vorherigen Antwort: {last}"
        if lang == "tl":
            return f"Oo. Batay sa naunang sagot: {last}"
        return f"Yes. Based on the previous answer: {last}"

    @staticmethod
    def _is_out_of_scope(question: str, relevant_memory: str) -> bool:
        # Legacy deterministic helper retained for compatibility references.
        q = (question or "").strip().lower()
        if not q:
            return False
        if any(re.search(pattern, q) for pattern in OUT_OF_SCOPE_PATTERNS):
            return True
        if CascadeOrchestrator._contains_followup_marker(q) and relevant_memory:
            return False
        if CascadeOrchestrator._is_confirmation_followup(q) and relevant_memory:
            return False
        if any(re.match(pattern, q) for pattern in FALLBACK_GENERAL_PATTERNS):
            return False
        if CascadeOrchestrator._looks_academic(q):
            return False
        if re.search(r"\b(where\s+is|nasaan|asan|nasan|near|nearby|beside|route|path|directions|papunta|paano\s+pumunta)\b", q):
            return False
        return bool(re.search(r"\b(who\s+is|what\s+is|where\s+is|when\s+is|why\s+is|how\s+to)\b", q))

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
    def _looks_low_quality_answer(question: str, answer: str) -> bool:
        q = (question or "").strip().lower()
        a = (answer or "").strip()
        if not a:
            return True

        # Common truncation patterns from OCR/LLM fragments.
        if re.search(r"\([a-zA-Z]?\.?$", a) or a.endswith(":"):
            return True

        # Very noisy OCR-like strings should not be preferred over cleaner fallback extraction.
        noise_hits = 0
        noise_hits += 1 if re.search(r"\b\d+\s*\|\s*Page\b", a, flags=re.IGNORECASE) else 0
        noise_hits += 1 if re.search(r"\b\d+\.\d+\.\d+\b", a) else 0
        noise_hits += 1 if re.search(r"\b(series|executive order|omnibus rules)\b", a, flags=re.IGNORECASE) else 0
        if noise_hits >= 2:
            return True

        if q.startswith("yes or no") or q.startswith("so yes or no"):
            lead = a.lower().strip()
            if not (lead.startswith("yes") or lead.startswith("no")):
                return True

        if q.startswith(("what is", "define", "meaning of")):
            if a.lower().startswith("definition of terms"):
                return True

        if (q.startswith("who is") or q.startswith("sino")) and "dean" in q:
            has_name = bool(re.search(r"\b(?:Dr|Mr|Ms|Engr)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\b", a))
            if not has_name and a.strip().startswith("-"):
                return True
            if re.search(r"\b(is|ay)\s+(cict|bulsu|college|office)\b", a, flags=re.IGNORECASE):
                return True

        if q.startswith("who is") or q.startswith("sino"):
            has_person = CascadeOrchestrator._has_likely_person_name(a)
            if re.search(r"\b(serves\s+as|shall\s+designate|prime\s+duty|educational\s+leader)\b", a, flags=re.IGNORECASE):
                return True
            if not has_person:
                return True

        return False

    @staticmethod
    def _has_likely_person_name(text: str) -> bool:
        content = str(text or "")
        if re.search(r"\b(?:Dr|Mr|Ms|Engr)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,5}\b", content):
            return True

        # Accept two to four capitalized tokens if they are not purely role labels.
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", content):
            candidate = m.group(1)
            if re.search(r"\b(Dean|Vice|Associate|Assistant|Office|President|Academic|Affairs|College|University|CICT|BulSU)\b", candidate):
                continue
            return True
        return False

    @staticmethod
    def _is_role_query(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        asks_person = bool(re.search(r"\b(who\s+is|sino\s+si|sino\s+ang|name\s+of)\b", q))
        mentions_role = bool(re.search(r"\b(dean|vice\s+dean|associate\s+dean|coordinator|director|chair|adviser|advisor)\b", q))
        return asks_person and mentions_role

    @staticmethod
    def _looks_noisy_sentence(sentence: str) -> bool:
        s = (sentence or "").strip()
        if len(s) < 20:
            return True
        if len(s) > 380:
            return True
        if re.search(r"\b\d+\s*\|\s*Page\b", s, flags=re.IGNORECASE):
            return True
        if re.search(r"\b\d+\.\d+\.\d+\b", s):
            return True
        alnum = re.findall(r"[A-Za-z0-9]", s)
        if not alnum:
            return True
        digits = re.findall(r"[0-9]", s)
        if len(digits) / max(len(alnum), 1) > 0.18:
            return True
        sym = re.findall(r"[^A-Za-z0-9\s\.,;:!?()\-]", s)
        if len(sym) / max(len(alnum), 1) > 0.25:
            return True
        return False

    @staticmethod
    def _resolve_followup_question(question: str, relevant_memory: str) -> str:
        q = (question or "").strip()
        if not q:
            return q
        if not CascadeOrchestrator._contains_followup_marker(q):
            return q
        if not relevant_memory:
            return q

        if CascadeOrchestrator._is_confirmation_followup(q):
            return q

        user_lines = [ln for ln in relevant_memory.splitlines() if ln.startswith("User:")]
        if not user_lines:
            return q
        last_user = user_lines[-1].replace("User:", "", 1).strip()
        if not last_user:
            return q

        # Resolve pronoun-based location follow-ups to explicit landmark queries.
        if re.search(r"\b(where\s+is\s+it\s+near|what\s+is\s+it\s+near|near\s+to\s+it|nearby\s+to\s+it)\b", q, flags=re.IGNORECASE):
            last_loc = re.search(r"\b(?:where\s+is|nasaan\s+ang|nasaan|asan|nasan)\s+(.+)$", last_user, flags=re.IGNORECASE)
            if last_loc:
                target = last_loc.group(1).strip(" ?!.,")
                if target:
                    return f"What is near {target}?"

        # For short follow-ups like "yes or no?", anchor to previous question.
        if q.lower() in {
            "yes or no", "yes or no?", "so yes or no", "so yes or no?",
            "yes oder nicht", "yes oder nicht?", "ja oder nein", "ja oder nein?",
        }:
            return f"{last_user} Please answer only Yes or No and add one short reason."

        # If user asks about another grade as a follow-up, keep shift/GWA intent anchored.
        grade_match = re.search(r"\b(?:grade|gwa)\s*(?:of|is)?\s*([1-5](?:\.\d+)?)\b", q, flags=re.IGNORECASE)
        if grade_match:
            if re.search(r"\bshift|shiftee|eligible|qualify\b", last_user, flags=re.IGNORECASE):
                return f"Can I shift if my GWA is {grade_match.group(1)}?"

        # Handle "how about a grade of X" phrasing robustly.
        if q.lower().startswith("how about"):
            num = re.search(r"\b([1-5](?:\.\d+)?)\b", q)
            memory_scope = f"{last_user}\n{relevant_memory}"
            if num and re.search(r"\bshift|shiftee|eligible|qualify|gwa\b", memory_scope, flags=re.IGNORECASE):
                return f"Can I shift if my GWA is {num.group(1)}?"

        # Reframe "how about ..." follow-ups into explicit question form.
        m = re.search(r"^how\s+about\s+(.+)$", q, flags=re.IGNORECASE)
        if m:
            subject = (m.group(1) or "").strip(" ?.!")
            subject = re.sub(r"\b(naman|nga|po|tho|lang)\b", "", subject, flags=re.IGNORECASE)
            subject = re.sub(r"\s+", " ", subject).strip()
            if subject:
                return f"What is {subject}?"
        return f"{last_user}\nFollow-up: {q}"

    @staticmethod
    def _is_definition_query(question: str) -> bool:
        q = (question or "").strip().lower()
        return bool(re.search(r"\b(what\s+is|define|meaning\s+of)\b", q))

    @staticmethod
    def _definition_from_context_loose(question: str, rag_context: str) -> str:
        q = (question or "").strip().lower()
        flat = re.sub(r"\s+", " ", rag_context or "")
        if not q or not flat:
            return ""
        m_term = re.search(r"\b(?:what\s+is|define|meaning\s+of)\s+(?:a|an|the)?\s*([a-z][a-z\-\s]{2,40})\??", q)
        if not m_term:
            return ""
        term = m_term.group(1).strip()
        term_clean = re.sub(r"\b(naman|nga|po|tho|lang)\b", "", term, flags=re.IGNORECASE).strip()
        if not term_clean:
            return ""
        term_pat = re.escape(term_clean)
        m = re.search(rf"\b{term_pat}\s+is\s+(.{{20,260}}?)(?:\.\s+[A-Z]|$)", flat, flags=re.IGNORECASE)
        if not m:
            return ""
        meaning = m.group(1).strip().rstrip(". ")
        if re.search(r"\([a-z]$", meaning, flags=re.IGNORECASE):
            return ""
        return f"{term_clean.capitalize()} is {meaning}."

    @staticmethod
    def _definition_from_preview_text(question: str, preview_text: str) -> str:
        q = (question or "").lower()
        p = re.sub(r"\s+", " ", (preview_text or "")).strip()
        if not p:
            return ""
        for term in ("shiftee", "transferee", "returnee"):
            if term in q:
                m = re.search(rf"\b{term}\s+is\s+(.{{20,320}}?)(?:\.\s+[A-Z]|$)", p, flags=re.IGNORECASE)
                if m:
                    meaning = m.group(1).strip().rstrip(". ")
                    return f"A {term} is {meaning}."
        return ""

    @staticmethod
    def _strip_definition_preface(text: str) -> str:
        t = re.sub(r"\s+", " ", (text or "")).strip()
        m = re.search(r"\b([A-Za-z]{3,30})\s+is\s+(.{20,320}?)(?:\.\s+[A-Z]|$)", t, flags=re.IGNORECASE)
        if not m:
            return t
        term = m.group(1).strip()
        meaning = m.group(2).strip().rstrip(". ")
        article = "An" if term[:1].lower() in {"a", "e", "i", "o", "u"} else "A"
        return f"{article} {term.lower()} is {meaning}."

    @staticmethod
    def _is_yes_no_followup(question: str) -> bool:
        q = (question or "").strip().lower()
        return q in {
            "yes or no", "yes or no?", "so yes or no", "so yes or no?",
            "yes oder nicht", "yes oder nicht?", "ja oder nein", "ja oder nein?",
        }

    @staticmethod
    def _yes_no_from_memory(relevant_memory: str) -> str:
        if not relevant_memory:
            return ""
        assistant_lines = [ln for ln in relevant_memory.splitlines() if ln.startswith("Assistant:")]
        if not assistant_lines:
            return ""
        last = assistant_lines[-1].replace("Assistant:", "", 1).strip()
        if not last:
            return ""
        low = last.lower()
        if low.startswith("yes"):
            return "Yes. Based on the previous answer, you qualify."
        if low.startswith("no"):
            return "No. Based on the previous answer, you do not qualify."
        return ""

    @staticmethod
    def _yes_no_from_memory_lang(relevant_memory: str, question: str) -> str:
        base = CascadeOrchestrator._yes_no_from_memory(relevant_memory)
        if not base:
            return ""
        lang = CascadeOrchestrator._detect_language_hint(question)
        if lang == "de":
            if base.lower().startswith("yes"):
                return "Ja. Basierend auf der vorherigen Antwort erfuellst du die Voraussetzung."
            return "Nein. Basierend auf der vorherigen Antwort erfuellst du die Voraussetzung nicht."
        if lang == "fr":
            if base.lower().startswith("yes"):
                return "Oui. D'apres la reponse precedente, vous etes eligible."
            return "Non. D'apres la reponse precedente, vous n'etes pas eligible."
        if lang == "tl":
            if base.lower().startswith("yes"):
                return "Oo. Batay sa naunang sagot, qualified ka."
            return "Hindi. Batay sa naunang sagot, hindi ka qualified."
        return base

    @staticmethod
    def _mission_vision_fallback(question: str, rag_context: str) -> str:
        q = (question or "").lower()
        if "mission" not in q or "vision" not in q:
            return ""
        cict_vision = (
            "Excellence in producing globally competitive graduates in the field of "
            "Information and Communications Technology responsive to the changing needs of society."
        )
        cict_mission = (
            "To provide quality education by ensuring efficient and effective delivery of instruction "
            "through appropriate adoption of technological innovation and research in carrying out extension services."
        )
        bulsu_vision = (
            "Bulacan State University is a progressive knowledge-generating institution globally recognized "
            "for excellent instruction, pioneering research, and responsive community engagements."
        )
        bulsu_mission = (
            "Bulacan State University exists to produce highly competent, ethical, and service-oriented professionals "
            "that contribute to the sustainable socio-economic growth and development of the nation."
        )

        if "cict" in q:
            return "\n".join(
                [
                    "CICT Vision and Mission:",
                    f"- Vision: {cict_vision}",
                    f"- Mission: {cict_mission}",
                ]
            )

        if "bulsu" in q or "bulacan state university" in q or "university" in q:
            return "\n".join(
                [
                    "BulSU Vision and Mission:",
                    f"- Vision: {bulsu_vision}",
                    f"- Mission: {bulsu_mission}",
                ]
            )

        return "\n".join(
            [
                "Here are both references for clarity:",
                f"- CICT Vision: {cict_vision}",
                f"- CICT Mission: {cict_mission}",
                f"- BulSU Vision: {bulsu_vision}",
                f"- BulSU Mission: {bulsu_mission}",
            ]
        )

    @staticmethod
    def _clean_person_name(candidate: str) -> str:
        raw = re.sub(r"\s+", " ", (candidate or "")).strip(" .,-")
        raw = re.sub(r"\b\S+\.pdf\b", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b(source|from|file)\b", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b(Associate|Assistant|Acting|Officer|Office|College|University|CICT|BulSU)\b", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s+", " ", raw).strip(" .,-")
        if not raw:
            return ""
        tokens = raw.split()
        if len(tokens) < 2:
            return ""
        if all(t.isupper() and len(t) <= 5 for t in tokens):
            return ""
        if any(t.lower() in {"cict", "bulsu", "college", "office"} for t in tokens):
            return ""
        return raw

    @staticmethod
    def _contains_followup_marker(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False

        phrase_markers = [
            "what about", "so ibig", "ibig sabihin", "pwede pa", "pano naman", "how about", "so yes", "so no",
            "are you sure", "u sure", "sure tho", "sigurado ka", "yes oder nicht", "ja oder nein",
            "how many minutes", "how mins", "how long", "ilang minuto", "gaano katagal",
        ]
        if any(m in q for m in phrase_markers):
            return True

        if q in {
            "yes or no", "yes or no?", "so yes or no", "so yes or no?",
            "yes oder nicht", "yes oder nicht?", "ja oder nein", "ja oder nein?",
        }:
            return True

        token_markers = {"only", "that", "those", "it", "they", "them", "her", "his", "she", "he", "there", "doon", "dito", "yun", "iyon", "ganon", "ganoon"}
        tokens = set(re.findall(r"[a-zA-Z0-9]+", q))
        return any(marker in tokens for marker in token_markers)

    @staticmethod
    def _is_travel_time_followup(question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        return bool(
            re.search(
                r"\b(how\s+many\s+minutes?|how\s+mins?|how\s+long|ilang\s+minuto|gaano\s+katagal|estimated\s+time|travel\s+time)\b",
                q,
            )
        )

    @staticmethod
    def _estimate_minutes_from_memory(relevant_memory: str) -> str:
        if not relevant_memory:
            return ""

        text = str(relevant_memory)
        cost_hits = list(re.finditer(r"Estimated\s+path\s+cost:\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE))
        if not cost_hits:
            return ""
        m_cost = cost_hits[-1]

        try:
            cost = float(m_cost.group(1))
        except Exception:
            return ""

        low_min = max(1, int(round(cost * 2.0)))
        high_min = max(low_min, int(round(cost * 4.0)))

        pair_match = re.search(r"between\s+(.+?)\s+and\s+(.+?):", text, flags=re.IGNORECASE)
        pair_text = ""
        if pair_match:
            pair_text = f" for {pair_match.group(1).strip()} to {pair_match.group(2).strip()}"

        return (
            f"Based on the latest mapped route{pair_text}, the walking estimate is about {low_min}-{high_min} minutes. "
            f"(Path cost: {cost})"
        )

    @staticmethod
    def _direct_fact_fallback(question: str, rag_context: str) -> str:
        q = (question or "").lower()
        flat = re.sub(r"\s+", " ", rag_context or "")
        context_lines = [
            ln.strip()
            for ln in (rag_context or "").splitlines()
            if ln.strip() and not ln.strip().lower().startswith("source:") and ln.strip() != "---"
        ]
        flat_no_sources = re.sub(r"\s+", " ", " ".join(context_lines))
        normalized_flat = unicodedata.normalize("NFKD", flat)
        normalized_flat_no_sources = unicodedata.normalize("NFKD", flat_no_sources)
        if not flat:
            return ""

        who_match = re.search(r"\b(?:who\s+is|sino\s+si|sino\s+ang)\s+(?:the\s+)?([a-z][a-z\s\-/]{2,60})\??", q)
        if who_match:
            role = who_match.group(1).strip().strip("?.")
            role = re.sub(r"\s+", " ", role)
            role = role.replace("assoc ", "associate ")
            role = re.sub(r"\b(of|ng|sa)\s+(the\s+)?cict\b", "", role, flags=re.IGNORECASE).strip()
            role = re.sub(r"\s+", " ", role)

            honorific_name = r"((?:Dr|Mr|Ms|Engr)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.]+){0,6}(?:\s+[IVX]+)?(?:,\s*[A-Z]{2,6})?)"
            if role == "associate dean":
                m_assoc = re.search(
                    rf"{honorific_name}\s*[^A-Za-z]{{0,10}}\s*Associate\s+Dean\s*,?\s*CICT",
                    normalized_flat_no_sources,
                    flags=re.IGNORECASE,
                )
                if m_assoc:
                    name = CascadeOrchestrator._clean_person_name(m_assoc.group(1) or "")
                    if name:
                        return f"The associate dean is {name}."
                # Avoid hallucinating dean/policy snippets when no associate-dean name is readable.
                return "I could not find a clear associate dean name in the current documents."

            if role in {"vice dean", "deputy dean"}:
                m_vice = re.search(
                    rf"{honorific_name}\s*[^A-Za-z]{{0,10}}\s*(?:Vice|Deputy)\s+Dean\s*,?\s*CICT",
                    normalized_flat_no_sources,
                    flags=re.IGNORECASE,
                )
                if m_vice:
                    name = CascadeOrchestrator._clean_person_name(m_vice.group(1) or "")
                    if name:
                        return f"The vice dean is {name}."

                # Some documents use Associate Dean instead of Vice Dean.
                m_assoc_alt = re.search(
                    rf"{honorific_name}\s*[^A-Za-z]{{0,10}}\s*Associate\s+Dean\s*,?\s*CICT",
                    normalized_flat_no_sources,
                    flags=re.IGNORECASE,
                )
                if m_assoc_alt:
                    name = CascadeOrchestrator._clean_person_name(m_assoc_alt.group(1) or "")
                    if name:
                        return (
                            f"I could not find a separate Vice Dean entry, but the Associate Dean listed is {name}."
                        )

                return "I could not find a clear vice dean name in the current documents."

            if role == "dean":
                m_dean = re.search(
                    rf"{honorific_name}\s*[^A-Za-z]{{0,10}}\s*Dean\s*,?\s*CICT",
                    normalized_flat_no_sources,
                    flags=re.IGNORECASE,
                )
                if m_dean:
                    name = CascadeOrchestrator._clean_person_name(m_dean.group(1) or "")
                    if name:
                        return f"The dean is {name}."
                m_surname = re.search(r"\b([A-Z][a-z]{2,30})\s+Dean\s*,?\s*CICT\b", normalized_flat_no_sources)
                if m_surname:
                    return f"The dean is {m_surname.group(1)}."

            role_variants = [role]
            if role == "dean":
                role_variants.extend(["dean, cict", "cict dean"])
            if role in {"associate dean", "assoc dean"}:
                role = "associate dean"
                role_variants = ["associate dean", "assoc dean", "associate dean, cict"]

            for rv in role_variants:
                role_pat = re.escape(rv).replace("\\ ", r"\\s+")
                m = re.search(rf"([A-Z][A-Za-z\.\s,IVX]{{2,100}})\s*[\-–|,:]?\s*{role_pat}\b", normalized_flat_no_sources, flags=re.IGNORECASE)
                if not m:
                    m = re.search(rf"{role_pat}\b\s*[\-–|,:]?\s*([A-Z][A-Za-z\.\s,IVX]{{2,100}})", normalized_flat_no_sources, flags=re.IGNORECASE)
                if m:
                    name = CascadeOrchestrator._clean_person_name(m.group(1) or "")
                    if name and not re.search(r"\b(page|rule|guideline|office|designation|vice presidents)\b", name, flags=re.IGNORECASE):
                        return f"The {role} is {name}."

        term_match = re.search(r"\b(?:what\s+is|define|meaning\s+of)\s+(?:a|an|the)?\s*([a-z][a-z\-\s]{2,40})\??", q)
        if term_match:
            term = re.escape(term_match.group(1).strip())
            m = re.search(rf"\b{term}\s+is\s+(.{{20,320}}?)(?:\.\s+[A-Z]|$)", flat, flags=re.IGNORECASE)
            if m:
                clean_term = term_match.group(1).strip()
                meaning = m.group(1).strip().rstrip(". ")
                if re.search(r"\([a-z]\.\s*$", meaning, flags=re.IGNORECASE):
                    return ""
                return f"{clean_term.capitalize()} is {meaning}."

        if "transferee" in q:
            m = re.search(r"Transferee\s+is\s+(.{20,320}?)(?:\.\s+[A-Z]|$)", flat, flags=re.IGNORECASE)
            if m:
                return f"A transferee is {m.group(1).strip().rstrip('. ')}."

        if "shiftee" in q:
            m = re.search(r"Shiftee\s+is\s+(.{20,320}?)(?:\.\s+[A-Z]|$)", flat, flags=re.IGNORECASE)
            if m:
                meaning = m.group(1).strip().rstrip('. ')
                if re.search(r"\([a-z]\.\s*$", meaning, flags=re.IGNORECASE):
                    return ""
                return f"A shiftee is {meaning}."

        if "returnee" in q:
            m = re.search(r"Returnee\s+is\s+(.{20,320}?)(?:\.\s+[A-Z]|$)", flat, flags=re.IGNORECASE)
            if m:
                return f"A returnee is {m.group(1).strip().rstrip('. ')}."

        if ("gwa" in q or re.search(r"\b\d+(?:\.\d+)?\b", q)) and any(k in q for k in {"shift", "shiftee", "qualify", "eligible", "can i"}):
            req = re.search(r"at\s+least\s+(?:a\s+)?gwa\s+of\s*([1-5](?:\s*[\.,]\s*\d+)?)", flat, flags=re.IGNORECASE)
            if not req:
                req = re.search(r"at\s+least\s*([1-5](?:\s*[\.,]\s*\d+)?)\s*gwa", flat, flags=re.IGNORECASE)
            actual = re.search(r"\b([1-5](?:\.\d+)?)\b", q)
            if req and actual:
                try:
                    req_num = re.sub(r"\s+", "", req.group(1)).replace(",", ".")
                    req_val = float(req_num)
                    actual_val = float(actual.group(1))
                    if actual_val <= req_val:
                        return (
                            f"Yes. Based on the stated rule of at least {req_val:.2f} GWA, a GWA of {actual_val:.2f} qualifies. "
                            "In BulSU grading, a lower number is better."
                        )
                    return (
                        f"No. Based on the stated rule of at least {req_val:.2f} GWA, a GWA of {actual_val:.2f} does not qualify. "
                        "In BulSU grading, a lower number is better."
                    )
                except Exception:
                    pass

        dean_name_query = (
            (("dean" in q and ("name" in q or "who" in q)) or ("her name" in q) or ("his name" in q))
            and ("associate dean" not in q)
            and ("vice dean" not in q)
            and ("deputy dean" not in q)
        )
        if dean_name_query:
            m = re.search(r"(Dr\.\s+[A-Z][A-Za-z\.\s]{2,60}?)\s*(?:-|,)?\s*Dean\b", flat)
            if not m:
                # OCR may drop punctuation around role labels.
                m = re.search(r"Dean\s*,?\s*CICT\s*[\.|,\-]?\s*(Dr\.\s+[A-Z][A-Za-z\.\s]{2,60})", flat)
            if not m:
                m = re.search(r"([A-Z][A-Za-z\.\s]{4,60})\s*[\-–|,:]\s*Dean\s*,?\s*CICT", flat, flags=re.IGNORECASE)
            if m:
                name = (m.group(1) or "").strip()
                name = re.sub(r"\s+", " ", name)
                return f"The CICT Dean is {name}."

    @staticmethod
    def _definition_keyword_fallback(question: str, rag_context: str) -> str:
        q = (question or "").lower()
        if not q:
            return ""
        context = re.sub(r"\s+", " ", rag_context or "").strip()
        if not context:
            return ""

        def _extract(term: str, article: str) -> str:
            patterns = [
                rf"\b{term}\b\s*(?:is|refers\s+to|means)\s+(.{{20,280}}?)(?:\.|;|\s+[A-Z][a-z]+\s*[:\-]|$)",
                rf"\b{term}\b\s*[:\-]\s*(.{{20,280}}?)(?:\.|;|\s+[A-Z][a-z]+\s*[:\-]|$)",
            ]
            for pat in patterns:
                m = re.search(pat, context, flags=re.IGNORECASE)
                if m:
                    meaning = (m.group(1) or "").strip().rstrip(" .")
                    if meaning:
                        return f"{article} {term} is {meaning}."
            return ""

        if "shiftee" in q:
            return _extract("shiftee", "A")
        if "transferee" in q:
            return _extract("transferee", "A")
        if "returnee" in q:
            return _extract("returnee", "A")
        if re.search(r"\btenure|tenured\b", q):
            hit = _extract("tenure", "Tenure") or _extract("tenured", "Tenured status")
            if hit:
                hit = re.sub(r"^Tenured status\s+tenured\s+is\s+", "Tenure means ", hit, flags=re.IGNORECASE)
                hit = re.sub(r"^Tenure\s+tenure\s+is\s+", "Tenure means ", hit, flags=re.IGNORECASE)
                return hit
        return ""

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
                    return "CICT BSIT Program Coordinators:\n" + "\n".join(lines[:5])

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
                if CascadeOrchestrator._looks_noisy_sentence(sent_clean):
                    continue
                s_lower = sent_clean.lower()
                score = sum(1 for term in q_terms if term in s_lower)
                q_lower = (question or "").lower()
                if ("associate" in q_lower or "assoc" in q_lower) and not ("associate" in s_lower or "assoc" in s_lower):
                    continue
                if "dean" in q_lower and "dean" not in s_lower:
                    continue
                if score > 0 and (question or "").lower().startswith(("who is", "sino")):
                    if re.search(r"\b(dean|associate dean|coordinator|chair|director|adviser|advisor)\b", s_lower):
                        score += 3
                    if re.search(r"\b(?:dr|mr|ms|engr)\.?\s+[a-z]", s_lower):
                        score += 2
                if score > 0:
                    scored.append((score, source, sent_clean))

        if not scored:
            return ""

        scored.sort(key=lambda item: item[0], reverse=True)
        unique_sentences = []
        seen = set()
        for _, _, sentence in scored:
            key = re.sub(r"\s+", " ", sentence.lower()).strip(" .")
            if key in seen:
                continue
            seen.add(key)
            unique_sentences.append(sentence)
            if len(unique_sentences) >= max_sentences:
                break

        if not unique_sentences:
            return ""

        q_lower = (question or "").strip().lower()
        if q_lower.startswith("what is") or q_lower.startswith("who is") or q_lower.startswith("sino"):
            if "definition of terms" in unique_sentences[0].lower():
                return CascadeOrchestrator._strip_definition_preface(unique_sentences[0])
            return unique_sentences[0]

        lines = [f"- {sentence}" for sentence in unique_sentences]
        return "\n".join(lines)

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
        candidates: List[Tuple[int, int, int, str]] = []
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

            if overlap >= CONFIG.memory_min_score or (followup and idx >= len(recent) - 4):
                assistant_text = ""
                if idx + 1 < len(recent) and recent[idx + 1].get("role") == "assistant":
                    assistant_text = (recent[idx + 1].get("content") or "").strip()
                block = f"User: {user_text}"
                if assistant_text:
                    block += f"\nAssistant: {assistant_text}"
                candidates.append((score, overlap, idx, block))

        if not candidates:
            return ""

        if followup:
            strongest_overlap = max((item[1] for item in candidates), default=0)
            if strongest_overlap == 0:
                latest = max(candidates, key=lambda item: item[2])
                return latest[3]

        picked = sorted(candidates, key=lambda item: item[0], reverse=True)[: CONFIG.memory_top_k]
        picked.sort(key=lambda item: item[2])
        return "\n\n".join(block for _, _, _, block in picked)

    async def respond(self, question: str, history: List[Dict]) -> Tuple[CascadeResult, str]:
        normalized_question = self._normalize_question(question)
        print(f"[Query] {question}", flush=True)
        if normalized_question != question:
            print(f"[Query] normalized -> {normalized_question}", flush=True)

        safety = await AdvancedSecurityGuard.validate_message(normalized_question, self.llm)
        if safety.get("block"):
            return CascadeResult(
                reply=AdvancedSecurityGuard.get_security_response(safety.get("category", "unknown")),
                route="SAFETY_BLOCK",
                context=safety.get("reason", "blocked"),
            ), safety.get("reason", "blocked")

        history_text = self._history_to_text(history)
        relevant_memory = self._relevant_memory(normalized_question, history)
        if not relevant_memory and self._is_known_place_phrase(normalized_question):
            relevant_memory = self._latest_exchange_from_history(history)
        resolved_question = self._resolve_followup_question(normalized_question, relevant_memory)
        resolved_question = self._resolve_location_only_followup(resolved_question, relevant_memory)

        route = await self._route(normalized_question)
        if route == "GENERAL" and resolved_question != normalized_question:
            route = "ACADEMIC"
        print(f"[Router] -> {route}", flush=True)
        if relevant_memory:
            print("[Memory] Relevant history found for current query", flush=True)
        if resolved_question != normalized_question:
            print(f"[Query] resolved follow-up -> {resolved_question}", flush=True)

        if self._is_confirmation_followup(normalized_question):
            confirm = self._confirmation_from_memory(relevant_memory, normalized_question)
            if confirm:
                return CascadeResult(reply=confirm, route="MEMORY_CONFIRM", context=""), "ok"
            latest_assistant = self._latest_assistant_from_history(history)
            if latest_assistant:
                confirm = self._confirmation_from_memory(f"Assistant: {latest_assistant}", normalized_question)
                if confirm:
                    return CascadeResult(reply=confirm, route="MEMORY_CONFIRM", context=""), "ok"
            return CascadeResult(reply=self._confirmation_without_memory(normalized_question), route="CONFIRM_CLARIFY", context=""), "ok"

        if self._is_yes_no_followup(normalized_question):
            memory_yes_no = self._yes_no_from_memory_lang(relevant_memory, normalized_question)
            if memory_yes_no:
                return CascadeResult(reply=memory_yes_no, route="MEMORY_FOLLOWUP", context=""), "ok"
            latest_assistant = self._latest_assistant_from_history(history)
            if latest_assistant:
                quick_yes_no = self._yes_no_from_memory_lang(f"Assistant: {latest_assistant}", normalized_question)
                if quick_yes_no:
                    return CascadeResult(reply=quick_yes_no, route="MEMORY_FOLLOWUP", context=""), "ok"

        if self._is_travel_time_followup(normalized_question):
            from_memory = self._estimate_minutes_from_memory(relevant_memory)
            if from_memory:
                return CascadeResult(reply=from_memory, route="SPATIAL_TIME_FOLLOWUP", context="memory_route"), "ok"
            latest_assistant = self._latest_assistant_from_history(history)
            if latest_assistant:
                quick_time = self._estimate_minutes_from_memory(f"Assistant: {latest_assistant}")
                if quick_time:
                    return CascadeResult(reply=quick_time, route="SPATIAL_TIME_FOLLOWUP", context="memory_route"), "ok"

        if route == "GENERAL":
            final = self._deterministic_general_reply(normalized_question)
            final = ResponseFilter.filter_response(final)
            validate = validate_model_response(final)
            if validate.blocked:
                final = "I can only provide safe BulSU CICT assistance."
            return CascadeResult(reply=final, route="GENERAL", context=""), "ok"

        # Block clearly unrelated topics early.
        if await self._is_out_of_scope_hybrid(normalized_question, relevant_memory):
            return CascadeResult(
                reply="I can only answer queries related to BulSU CICT.",
                route="OUT_OF_SCOPE",
                context="",
            ), "ok"

        query_variants = await self._prepare_query_variants(resolved_question)
        corrected_query = query_variants.get("corrected_query", "")
        english_query = query_variants.get("english_query", "")
        slang_expansion = query_variants.get("slang_expansion", "")

        spatial_probe = resolved_question or corrected_query
        spatial_reply = self.spatial_tool.run(spatial_probe)
        if spatial_reply:
            print("[Tool] Spatial tool handled query", flush=True)
            return CascadeResult(reply=spatial_reply, route="SPATIAL_TOOL", context="spatial_graph"), "ok"

        # Cascading workflow: GENERAL -> RAG -> WEB
        print("[Cascade] Trying RAG", flush=True)
        retrieval_query = self._coalesce_queries(resolved_question, corrected_query, slang_expansion, english_query)
        retrieval_query = self._expand_room_tokens(retrieval_query)
        rq_lower = resolved_question.lower()
        if "gwa" in rq_lower and any(k in rq_lower for k in {"shift", "shiftee", "qualify", "eligible", "can i"}):
            retrieval_query += "\n\ngrade requirements at least gwa shiftee shifting eligibility"
        if self._contains_followup_marker(normalized_question) and relevant_memory:
            user_lines = [ln for ln in relevant_memory.splitlines() if ln.startswith("User:")]
            if user_lines:
                last_user = user_lines[-1].replace("User:", "", 1).strip()
                if last_user:
                    retrieval_query = f"{retrieval_query}\n\nFollow-up context: {last_user}"

        rag_context_candidates: List[str] = []
        for probe in [retrieval_query, corrected_query, english_query]:
            probe_query = (probe or "").strip()
            if not probe_query:
                continue
            ctx = self.rag.context_for(probe_query)
            if ctx:
                rag_context_candidates.append(ctx)
        rag_context = self._merge_contexts(rag_context_candidates, max_blocks=max(8, CONFIG.rag_rerank_top_n + 3))
        rag_chunks = rag_context.count("Source:") if rag_context else 0
        print(f"[Cascade] RAG chunks={rag_chunks}", flush=True)
        loc_query = any(k in rq_lower for k in {"where", "saan", "campus", "location", "located"})
        context_relevant = await self._is_context_relevant(resolved_question, rag_context) if rag_context else False
        if rag_context and loc_query:
            context_relevant = True
        if rag_context:
            print(f"[Cascade] RAG relevance={'yes' if context_relevant else 'no'}", flush=True)
        rag_preview = ""
        if rag_context and context_relevant:
            mv_direct = self._mission_vision_fallback(resolved_question, rag_context)
            if mv_direct and not self._looks_low_quality_answer(normalized_question, mv_direct):
                return CascadeResult(reply=mv_direct, route="RAG_DIRECT_MV", context=rag_context), "ok"

            if self._is_role_query(resolved_question):
                role_direct = self._direct_fact_fallback(resolved_question, rag_context)
                if role_direct:
                    return CascadeResult(reply=role_direct, route="RAG_DIRECT_ROLE", context=rag_context), "ok"

            rag_user_prompt = f"Current Question: {resolved_question}"
            if relevant_memory:
                rag_user_prompt = (
                    "Context from previous conversation (only use if necessary to understand the current question):\n"
                    f"{relevant_memory}\n\n"
                    f"Current Question: {resolved_question}"
                )
            rag_answer = await self.llm.chat_with_fallbacks(
                rag_system_prompt(rag_context),
                rag_user_prompt,
            )

            # Prefer concise deterministic extraction for definition-style queries.
            if self._is_definition_query(resolved_question):
                direct_def = self._direct_fact_fallback(resolved_question, rag_context)
                if direct_def and not self._looks_low_quality_answer(normalized_question, direct_def):
                    return CascadeResult(reply=direct_def, route="RAG_DIRECT", context=rag_context), "ok"

            if rag_answer and "__NO_KB_ANSWER__" not in rag_answer:
                rag_answer = ResponseFilter.filter_response(rag_answer)
                validate = validate_model_response(rag_answer)
                if not validate.blocked and not self._looks_low_quality_answer(normalized_question, rag_answer):
                    return CascadeResult(reply=rag_answer, route="RAG", context=rag_context), "ok"
                print("[Cascade] RAG answer quality low; trying fallback extraction", flush=True)

            direct = self._direct_fact_fallback(resolved_question, rag_context)
            if direct and not self._looks_low_quality_answer(normalized_question, direct):
                return CascadeResult(reply=direct, route="RAG_DIRECT", context=rag_context), "ok"
            if direct:
                print("[Cascade] Direct fallback quality low; trying extractive fallback", flush=True)

            extractive = self._extractive_rag_fallback(resolved_question, rag_context)
            if extractive and not self._looks_low_quality_answer(normalized_question, extractive):
                return CascadeResult(reply=extractive, route="RAG_EXTRACTIVE", context=rag_context), "ok"

            rag_preview = self._preview_from_rag_context(rag_context)
            if not rag_answer:
                print("[Cascade] RAG answer empty; trying WEB before fallback preview", flush=True)
        elif rag_context and not context_relevant:
            print("[Cascade] RAG context deemed not relevant; trying direct fallback only", flush=True)
            mv_direct = self._mission_vision_fallback(resolved_question, rag_context)
            if mv_direct and not self._looks_low_quality_answer(normalized_question, mv_direct):
                return CascadeResult(reply=mv_direct, route="RAG_DIRECT_MV", context=rag_context), "ok"

            if self._is_role_query(resolved_question):
                role_direct = self._direct_fact_fallback(resolved_question, rag_context)
                if role_direct:
                    return CascadeResult(reply=role_direct, route="RAG_DIRECT_ROLE_LOWREL", context=rag_context), "ok"

            keyword_def = self._definition_keyword_fallback(resolved_question, rag_context)
            if keyword_def and not self._looks_low_quality_answer(normalized_question, keyword_def):
                return CascadeResult(reply=keyword_def, route="RAG_DIRECT_DEF_LOWREL", context=rag_context), "ok"

            direct = self._direct_fact_fallback(resolved_question, rag_context)
            if direct and not self._looks_low_quality_answer(normalized_question, direct):
                return CascadeResult(reply=direct, route="RAG_DIRECT_LOWREL", context=rag_context), "ok"

            extractive = self._extractive_rag_fallback(resolved_question, rag_context)
            if extractive and not self._looks_low_quality_answer(normalized_question, extractive):
                return CascadeResult(reply=extractive, route="RAG_EXTRACTIVE_LOWREL", context=rag_context), "ok"

        print("[Cascade] Trying WEB", flush=True)
        web_query = corrected_query or english_query or resolved_question
        web_context = await self.web.search(web_query)
        web_hits = web_context.count("From https://") if web_context else 0
        print(f"[Cascade] WEB hits={web_hits}", flush=True)
        if web_context:
            web_user_prompt = f"Question: {resolved_question}"
            if relevant_memory:
                web_user_prompt = (
                    "Relevant conversation memory:\n"
                    f"{relevant_memory}\n\n"
                    f"Question: {resolved_question}"
                )
            web_answer = await self.llm.chat_with_fallbacks(
                web_system_prompt(web_context),
                web_user_prompt,
            )
            if web_answer and "__NO_WEB_ANSWER__" not in web_answer:
                web_answer = ResponseFilter.filter_response(web_answer)
                validate = validate_model_response(web_answer)
                if not validate.blocked and not self._looks_low_quality_answer(normalized_question, web_answer):
                    return CascadeResult(reply=web_answer, route="WEB", context=web_context), "ok"

        if rag_preview:
            if self._is_definition_query(resolved_question):
                loose_def = self._definition_from_context_loose(resolved_question, rag_context)
                if loose_def and not self._looks_low_quality_answer(normalized_question, loose_def):
                    return CascadeResult(reply=loose_def, route="RAG_DIRECT_LOOSE", context=rag_context), "ok"
                preview_def = self._definition_from_preview_text(resolved_question, rag_preview)
                if preview_def and not self._looks_low_quality_answer(normalized_question, preview_def):
                    return CascadeResult(reply=preview_def, route="RAG_DIRECT_PREVIEW", context=rag_context), "ok"
                if "definition of terms" in rag_preview.lower() and " is " in rag_preview.lower():
                    trimmed = self._strip_definition_preface(rag_preview)
                    if trimmed and not self._looks_low_quality_answer(normalized_question, trimmed):
                        return CascadeResult(reply=trimmed, route="RAG_DIRECT_TRIMMED", context=rag_context), "ok"
            if normalized_question.lower().startswith(("who is", "sino")):
                return CascadeResult(
                    reply="I could not find a clear person name for that role in the current documents.",
                    route="NO_CLEAR_ROLE_NAME",
                    context=rag_context,
                ), "ok"
            return CascadeResult(
                reply=(
                    "Here is the most relevant information I found:\n\n"
                    f"{rag_preview}"
                ),
                route="RAG_PREVIEW",
                context=rag_context,
            ), "ok"

        return CascadeResult(
            reply="I can only answer queries related to BulSU CICT.",
            route="NO_RESULT",
            context="",
        ), "ok"

    async def close(self) -> None:
        await self.llm.close()
