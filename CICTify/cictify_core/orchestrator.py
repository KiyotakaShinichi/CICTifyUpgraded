import asyncio
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
from .web_search import WebFallback


ROUTER_PROMPT = """You are a multilingual intent router for BulSU CICT assistant.

Task:
Classify the user query into exactly one label:
- GENERAL: greetings, identity, small talk, thanks, short chit-chat.
- ACADEMIC: any BulSU/CICT information request, policies, procedures, announcements, schedules, admissions, grading, room/location, requirements, offices, faculty, or factual queries.

Rules:
- Support multilingual input (English, Filipino/Tagalog, Taglish, French, Spanish, German, and similar variations).
- If the query is not clearly casual, choose ACADEMIC.
- Output exactly one token only: GENERAL or ACADEMIC.

Examples:
- "hi" -> GENERAL
- "huy" -> GENERAL
- "who are you" -> GENERAL
- "ano name mo" -> GENERAL
- "salamat" -> GENERAL
- "good morning" -> GENERAL
- "bonjour" -> GENERAL
- "hola" -> GENERAL
- "guten tag" -> GENERAL
- "ciao" -> GENERAL
- "ano requirements sa enrollment" -> ACADEMIC
- "where is the dean's office" -> ACADEMIC
- "paano mag shift to BSIT" -> ACADEMIC
- "sino dean ng cict" -> ACADEMIC
- "what is the grading policy" -> ACADEMIC
"""

FALLBACK_GENERAL_PATTERNS = [
    r"^(hi|hello|hey|heya|hii+|helo+|huy|hu|yo|sup|kumusta|kamusta|bonjour|hola|salut|ciao|guten\s+tag)\b",
    r"^good\s+(morning|afternoon|evening)\b",
    r"^(who\s+are\s+you|what'?s\s+your\s+name|sino\s+ka|ano\s+(pangalan|name)\s+mo|qui\s+es-tu|wer\s+bist\s+du|chi\s+sei)\b",
    r"^(where\s+are\s+you|nasaan\s+ka|nasan\s+ka|asan\s+ka|ou\s+es-tu|wo\s+bist\s+du|dove\s+sei)\b",
    r"^(what\s+languages?\s+do\s+you\s+speak|anong\s+wika|language\s+mo|quelles\s+langues|welche\s+sprachen|quali\s+lingue)\b",
    r"^(thanks|thank\s+you|salamat|ok|okay|sige|bye|goodbye|ingat|merci|danke|grazie)\b",
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

        if re.search(r"\b(thanks|thank\s+you|salamat|bye|goodbye|ingat)\b", q):
            return "You are welcome."

        return "I can help with BulSU CICT questions."

    @staticmethod
    def _normalize_question(question: str) -> str:
        q = (question or "").strip()
        replacements = {
            r"\bashiftee\b": "a shiftee",
            r"\btransferee\b": "transferee",
            r"\bdeen\b": "dean",
            r"\bhow\s+a\s*bout\b": "how about",
            r"\ba\s+bout\b": "about",
        }
        for pattern, repl in replacements.items():
            q = re.sub(pattern, repl, q, flags=re.IGNORECASE)
        return q

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

        if (q.startswith("who is") or q.startswith("sino")) and "dean" in q:
            has_name = bool(re.search(r"\b(?:Dr|Mr|Ms|Engr)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\b", a))
            if not has_name and a.strip().startswith("-"):
                return True
            if re.search(r"\b(is|ay)\s+(cict|bulsu|college|office)\b", a, flags=re.IGNORECASE):
                return True

        if q.startswith("who is") or q.startswith("sino"):
            has_person = bool(re.search(r"\b(?:Dr|Mr|Ms|Engr)\.?\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4}\b", a))
            has_person |= bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", a))
            if re.search(r"\b(serves\s+as|shall\s+designate|prime\s+duty|educational\s+leader)\b", a, flags=re.IGNORECASE):
                return True
            if not has_person:
                return True

        return False

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

        user_lines = [ln for ln in relevant_memory.splitlines() if ln.startswith("User:")]
        if not user_lines:
            return q
        last_user = user_lines[-1].replace("User:", "", 1).strip()
        if not last_user:
            return q

        # For short follow-ups like "yes or no?", anchor to previous question.
        if q.lower() in {"yes or no", "yes or no?", "so yes or no", "so yes or no?"}:
            return f"{last_user} Please answer only Yes or No and add one short reason."

        # Reframe "how about ..." follow-ups into explicit question form.
        m = re.search(r"^how\s+about\s+(.+)$", q, flags=re.IGNORECASE)
        if m:
            subject = (m.group(1) or "").strip(" ?.!")
            if subject:
                return f"What is {subject}?"
        return f"{last_user}\nFollow-up: {q}"

    @staticmethod
    def _is_yes_no_followup(question: str) -> bool:
        q = (question or "").strip().lower()
        return q in {"yes or no", "yes or no?", "so yes or no", "so yes or no?"}

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
    def _clean_person_name(candidate: str) -> str:
        raw = re.sub(r"\s+", " ", (candidate or "")).strip(" .,-")
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
        ]
        if any(m in q for m in phrase_markers):
            return True

        if q in {"yes or no", "yes or no?", "so yes or no", "so yes or no?"}:
            return True

        token_markers = {"only", "that", "those", "it", "they", "them", "her", "his", "she", "he", "yun", "iyon", "ganon", "ganoon"}
        tokens = set(re.findall(r"[a-zA-Z0-9]+", q))
        return any(marker in tokens for marker in token_markers)

    @staticmethod
    def _direct_fact_fallback(question: str, rag_context: str) -> str:
        q = (question or "").lower()
        flat = re.sub(r"\s+", " ", rag_context or "")
        normalized_flat = unicodedata.normalize("NFKD", flat)
        if not flat:
            return ""

        who_match = re.search(r"\b(?:who\s+is|sino\s+si|sino\s+ang)\s+(?:the\s+)?([a-z][a-z\s\-/]{2,60})\??", q)
        if who_match:
            role = who_match.group(1).strip().strip("?.")
            role = re.sub(r"\s+", " ", role)
            role = role.replace("assoc ", "associate ")
            role_pat = re.escape(role).replace("\\ ", r"\\s+")
            m = re.search(rf"([A-Z][A-Za-z\.\s,IVX]{{2,100}})\s*[\-–|,:]\s*{role_pat}\b", normalized_flat, flags=re.IGNORECASE)
            if not m:
                m = re.search(rf"{role_pat}\b\s*[\-–|,:]?\s*([A-Z][A-Za-z\.\s,IVX]{{2,100}})", normalized_flat, flags=re.IGNORECASE)
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

        dean_name_query = ("dean" in q and ("name" in q or "who" in q)) or ("her name" in q) or ("his name" in q)
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
        recent = history[-CONFIG.max_history_messages:]

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

        route = await self._route(normalized_question)
        print(f"[Router] -> {route}", flush=True)
        history_text = self._history_to_text(history)
        relevant_memory = self._relevant_memory(normalized_question, history)
        resolved_question = self._resolve_followup_question(normalized_question, relevant_memory)
        if relevant_memory:
            print("[Memory] Relevant history found for current query", flush=True)
        if resolved_question != normalized_question:
            print(f"[Query] resolved follow-up -> {resolved_question}", flush=True)

        if self._is_yes_no_followup(normalized_question):
            memory_yes_no = self._yes_no_from_memory(relevant_memory)
            if memory_yes_no:
                return CascadeResult(reply=memory_yes_no, route="MEMORY_FOLLOWUP", context=""), "ok"

        if route == "GENERAL":
            final = self._deterministic_general_reply(normalized_question)
            final = ResponseFilter.filter_response(final)
            validate = validate_model_response(final)
            if validate.blocked:
                final = "I can only provide safe BulSU CICT assistance."
            return CascadeResult(reply=final, route="GENERAL", context=""), "ok"

        spatial_reply = self.spatial_graph.answer_navigation_query(normalized_question)
        if spatial_reply:
            return CascadeResult(reply=spatial_reply, route="SPATIAL_RAG", context="spatial_graph"), "ok"

        # Cascading workflow: GENERAL -> RAG -> WEB
        print("[Cascade] Trying RAG", flush=True)
        retrieval_query = self._expand_room_tokens(resolved_question)
        rq_lower = resolved_question.lower()
        if "gwa" in rq_lower and any(k in rq_lower for k in {"shift", "shiftee", "qualify", "eligible", "can i"}):
            retrieval_query += "\n\ngrade requirements at least gwa shiftee shifting eligibility"
        if self._contains_followup_marker(normalized_question) and relevant_memory:
            user_lines = [ln for ln in relevant_memory.splitlines() if ln.startswith("User:")]
            if user_lines:
                last_user = user_lines[-1].replace("User:", "", 1).strip()
                if last_user:
                    retrieval_query = f"{retrieval_query}\n\nFollow-up context: {last_user}"

        rag_context = self.rag.context_for(retrieval_query)
        rag_chunks = rag_context.count("Source:") if rag_context else 0
        print(f"[Cascade] RAG chunks={rag_chunks}", flush=True)
        rag_preview = ""
        if rag_context:
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

        print("[Cascade] Trying WEB", flush=True)
        web_context = await self.web.search(resolved_question)
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
            reply="Wala sa knowledge ko ngayon ang sagot sa tanong na iyan based on my BulSU CICT knowledge base and prewarmed web sources.",
            route="NO_RESULT",
            context="",
        ), "ok"

    async def close(self) -> None:
        await self.llm.close()
