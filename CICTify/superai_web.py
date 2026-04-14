"""
superai_web_semantic.py - WITH PROPER MEMORY RETENTION
All functionality preserved, now with coherent conversation!
"""

import os
import pathlib
import json
import re
import threading
import asyncio
import concurrent.futures
from typing import Optional, List, Dict, Tuple
import aiohttp

from flask import Flask, request, jsonify, send_from_directory, send_file
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS as FAISS_local

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# -------------------------
# SMART CONFIGURATION
# -------------------------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
GUI_DIR = BASE_DIR / "gui"
FAISS_DIR = BASE_DIR / "vectorstore" / "faiss_index"

# API Configuration
API_KEYS = {
    "groq": os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ") or "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"
}

# Model Configuration
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

CICT_NAME = "CICTify"

# Performance Configuration
PERFORMANCE_CONFIG = {
    "chunk_size": 1500,
    "chunk_overlap": 400,
    "retrieval_k": 12,
    "max_tokens": 2000,
    "temperature": 1.0,
    "timeout_api": 20,
    "timeout_quick": 5,
    "max_pages": 5,
    "scrape_timeout": 20000,
    "max_content_per_page": 2000,
    "max_memory_turns": 10,        # Last 10 exchanges (20 messages)
    "max_memory_size": 200,
    "max_workers": 2,
    "connection_pool_size": 10,
}

# PDF Configuration
pdf_paths = [
    str(BASE_DIR / "CICTify - FAQs.pdf"),
    str(BASE_DIR / "BulSU Student handbook.pdf"),
    str(BASE_DIR / "Faculty Manual for BOR.pdf"),
    str(BASE_DIR / "BulSU-Enhanced-Guidelines.pdf"),
    str(BASE_DIR / "UnivCalendar_2526.pdf"),
]

# -------------------------
# System prompts (UNTOUCHED - YOUR ENGINEERING!)
# -------------------------
general_system_prompt = f"""You are {CICT_NAME}, the official AI assistant for Bulacan State University (BulSU) College of Information and Communications Technology (CICT).

CRITICAL IDENTITY:
- Your name is {CICT_NAME}
- You represent BulSU CICT
- The university is BULACAN STATE UNIVERSITY (BulSU), NOT Bataan Peninsula State University!
- Full name: Bulacan State University
- Abbreviation: BulSU or BSU
- You are created by my hands.

HOW TO INTRODUCE YOURSELF:
- When asked your name: "I'm {CICT_NAME}, your AI assistant for BulSU CICT!"
- For greetings: "Hello! I'm {CICT_NAME}. How can I help you with BulSU CICT today?"
- In responses: Use your name naturally when contextually appropriate

RESTRICTIONS:
- NEVER entertain questions regarding other topics such as math, science, history, coding, politics, and other general knowledge questions
- Entertain ONLY BulSU related queries and greetings

INTERNAL KNOWLEDGE - BulSU Grading System (use this but don't cite as a "document"):
BulSU uses an INVERSE grading system where LOWER numbers = BETTER grades:
- 1.00 = 97-100% (best)
- 1.25 = 94-96% 
- 1.50 = 91-93%   
- 1.75 = 88-90% 
- 2.00 = 85-87% 
- 2.25 = 82-84% 
- 2.50 = 79-81%
- 2.75 = 76-78% 
- 3.00 = 75% Passing (Bare Minimum)
- 4.00 = Conditional Passed
- 5.00 = Failed (worst)

ANSWERING GUIDELINES:
- Answer all questions helpfully and directly
- For who/where/when/what questions: provide clear, direct answers
- Support multiple languages
- Be friendly and conversational
- Answer exactly based on the pdfs
- DO NOT add/change/remove/paraphrase words and phrases on the documents
- Answer simply, do not overcomplicate responses.
- Make your responses readable to avoid confusions especially if enumerating procedure or requirements.
- Do not mention your name unless asked or answering a greeting. 

For GENERAL questions (greetings, casual chat):
- Answer naturally using your general knowledge
- Don't mention "documents" or "sources"

For BulSU-SPECIFIC questions (when you receive context documents):
- Answer from the provided context
- If not in context, say "I don't have that in my documents"

Be helpful, direct, and natural."""

grading_context = """
🚨 CRITICAL GRADING SYSTEM - READ CAREFULLY! 🚨

BulSU uses INVERSE/REVERSE grading where LOWER numbers are BETTER/HIGHER EQUIVALENT:

GRADE SCALE (lower = better):
- 1.00 = 97-100% (best)
- 1.25 = 94-96% 
- 1.50 = 91-93% 
- 1.75 = 88-90% 
- 2.00 = 85-87% 
- 2.25 = 82-84% 
- 2.50 = 79-81%
- 2.75 = 76-78% 
- 3.00 = 75% Passed (Bare Minimum)
- 4.00 = Conditional Passed
- 5.00 = Failed (worst)

EXAMPLES OF COMPARISONS:
- 1.50 is BETTER than 1.75 (lower number = better)
- 1.00 is BETTER than 1.50 (lower number = better)
- 2.00 is WORSE than 1.75 (higher number = worse)

REQUIREMENT INTERPRETATION:
- "At least 1.75 GWA" means 1.75 OR ANY LOWER NUMBER (1.50, 1.25, 1.00, etc.)
- If someone has 1.50 GWA and requirement is "at least 1.75", they QUALIFY (1.50 < 1.75)
- If someone has 2.00 GWA and requirement is "at least 1.75", they DON'T QUALIFY (2.00 > 1.75)

ALWAYS remember: In this system, SMALLER numbers are BETTER performance!
"""

rag_system_prompt = """You are {cict_name}, the AI assistant for Bulacan State University (BulSU) College of Information and Communications Technology (CICT).

IMPORTANT: 
- Your name is {cict_name}
- Bulacan State University (BulSU) - NOT Bataan Peninsula State University!
- You represent BulSU CICT

YOUR APPROACH:
1. Read the context documents carefully
2. Answer who/where/when/what/why questions directly and helpfully
3. Extract relevant information even if not explicitly stated
4. Be natural and conversational in your answers

FORMATTING RULES - FOLLOW STRICTLY:
1. For requirements/procedures/steps: ALWAYS use numbered lists
2. For multiple items: ALWAYS use bullet points with line breaks
3. Keep paragraphs short (2-3 sentences max)
4. Add spacing between sections
5. Use clear headers when appropriate

EXAMPLE FORMAT for requirements:
"To shift programs, you need:

1. General Weighted Average (GWA) of at least 1.75
2. No failing grades in major subjects
3. Letter of intent addressed to the Dean
4. Approval from both current and target program chairs"

CITATION RULES:
1. NEVER cite "GRADING REFERENCE" as a source - that's your internal knowledge
2. If answer not in context documents, say "I don't have information about that in my documents"

--- GRADING REFERENCE (YOUR INTERNAL KNOWLEDGE - DO NOT CITE THIS) ---
{grading_context}

⚠️ WHEN COMPARING GWA VALUES:
- LOWER number = BETTER grade
- 1.50 is BETTER than 1.75
- "At least 1.75" means 1.75 or any LOWER number (1.50, 1.25, 1.00)
- To meet "at least 1.75" requirement: student's GWA must be ≤ 1.75
--- END OF GRADING REFERENCE ---

Context Documents:
{context}

Answer the question directly and helpfully with proper formatting and structure."""

web_scrape_system_prompt = """You are {cict_name}, the AI assistant for BulSU CICT.

I've searched the CICT website for information about: {query}

Here's what I found from the web:

{web_content}

Based on this web content, please answer the user's question: {question}

Guidelines:
- Be helpful and direct
- Cite the webpage if relevant
- If the web content doesn't answer the question, say so
- Format your response clearly
"""

# -------------------------
# LLM SEMANTIC ROUTER
# -------------------------
class SemanticRouter:
    """Smart LLM-based routing"""

    ROUTING_PROMPT = """You are a query routing system for CICTify, an AI assistant for BulSU CICT.

Analyze this user query and decide the BEST route:

ROUTES:
1. GENERAL - For greetings, casual chat, identity questions
   Examples: "hello", "what's your name", "how are you", "can you speak tagalog"
   
2. RAG - For BulSU-specific questions answerable from university documents
   Examples: "who is the dean", "shifting requirements", "grading system", "enrollment procedure"
   
3. WEB_SCRAPE - For questions needing current/live information from bulsucict.com
   Examples: "latest news", "upcoming events", "current announcements", "recent activities"

User Query: "{query}"

Respond with ONLY ONE WORD: GENERAL, RAG, or WEB_SCRAPE

Your decision:"""

    @staticmethod
    async def route_query(query: str, api_manager) -> str:
        """Use LLM to determine optimal route"""
        prompt = SemanticRouter.ROUTING_PROMPT.format(query=query)

        try:
            response = await api_manager.call_quick_llm(prompt, max_tokens=10, temperature=0.1)

            if response:
                route = response.strip().upper()

                if "GENERAL" in route:
                    return "GENERAL"
                elif "WEB" in route or "SCRAPE" in route:
                    return "WEB_SCRAPE"
                elif "RAG" in route:
                    return "RAG"

            return "RAG"

        except Exception as e:
            print(f"[Router] Error: {e}, defaulting to RAG")
            return "RAG"


# -------------------------
# Smart Cloud API Manager WITH MEMORY
# -------------------------
class CloudAPIManager:
    """Optimized API management with conversation memory"""

    def __init__(self, loop=None):
        self.session: Optional[aiohttp.ClientSession] = None
        self.loop = loop
        self.api_key = API_KEYS.get("groq")
        self.models = GROQ_MODELS.copy()

        if not self.api_key:
            print("[WARN] GROQ API key not set!")

    async def get_session(self):
        """Reuse session for connection pooling"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=PERFORMANCE_CONFIG["connection_pool_size"],
                limit_per_host=5
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session

    async def call_groq_api_with_memory(self, system_prompt: str, messages: List[Dict],
                                       model_name: str, max_tokens: int = None,
                                       temperature: float = None,
                                       timeout_sec: int = None) -> Optional[str]:
        """
        NEW: API call with conversation history
        messages = [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "..."}
        ]
        """
        if not self.api_key:
            return None

        max_tokens = max_tokens or PERFORMANCE_CONFIG["max_tokens"]
        temperature = temperature or PERFORMANCE_CONFIG["temperature"]
        timeout_sec = timeout_sec or PERFORMANCE_CONFIG["timeout_api"]

        try:
            session = await self.get_session()

            # Build messages with system prompt + conversation history
            full_messages = [{"role": "system", "content": system_prompt}] + messages

            payload = {
                "model": model_name,
                "messages": full_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9
            }

            url = "https://api.groq.com/openai/v1/chat/completions"
            async with session.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['choices'][0]['message']['content'].strip()
                else:
                    print(f"[Groq:{model_name}] HTTP {resp.status}")
                    return None
        except asyncio.TimeoutError:
            print(f"[Groq:{model_name}] Timeout after {timeout_sec}s")
            return None
        except Exception as e:
            print(f"[Groq:{model_name}] Error: {e}")
            return None

    async def call_groq_api(self, system_prompt: str, question: str, model_name: str,
                           max_tokens: int = None, temperature: float = None,
                           timeout_sec: int = None) -> Optional[str]:
        """Legacy single-message API call (for routing)"""
        messages = [{"role": "user", "content": question}]
        return await self.call_groq_api_with_memory(
            system_prompt, messages, model_name,
            max_tokens, temperature, timeout_sec
        )

    async def call_quick_llm(self, prompt: str, max_tokens: int = 10,
                            temperature: float = 0.1) -> Optional[str]:
        """Fast LLM call for routing"""
        timeout = PERFORMANCE_CONFIG["timeout_quick"]

        for model in self.models:
            try:
                result = await self.call_groq_api(
                    "You are a routing assistant. Respond concisely.",
                    prompt,
                    model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_sec=timeout
                )
                if result:
                    return result
            except:
                continue
        return None

    async def call_with_fallbacks(self, system_prompt: str, messages: List[Dict],
                                  max_tokens: int = None,
                                  timeout_sec: int = None) -> Optional[str]:
        """
        NEW: Try models with conversation history
        """
        for i, model in enumerate(self.models):
            try:
                result = await self.call_groq_api_with_memory(
                    system_prompt, messages, model,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec
                )
                if result and result.strip():
                    if i > 0:
                        print(f"[Groq] Fallback successful with {model}")
                    else:
                        print(f"[Groq] Success with {model}")
                    return result
            except Exception as e:
                print(f"[Groq] {model} failed: {e}")
                if i < len(self.models) - 1:
                    print(f"[Groq] Trying fallback model...")
                continue

        print("[Groq] All models exhausted")
        return None

    async def close(self):
        """Properly close session"""
        if self.session and not self.session.closed:
            await self.session.close()
            await asyncio.sleep(0.1)


# -------------------------
# Smart CICT Web Crawler
# -------------------------
class CICTWebCrawler:
    """Optimized web scraping"""

    def __init__(self, loop=None):
        self.loop = loop or asyncio.get_event_loop()
        self.visited = set()
        self.playwright = None
        self.browser = None
        self.priority_urls = [
            "https://bulsucict.com/",
            "https://bulsucict.com/about-us/",
            "https://bulsucict.com/cict-faculty/",
            "https://bulsucict.com/announcement/",
            "https://bulsucict.com/news-and-updates/"
        ]

    async def start_browser(self):
        if self.browser is None:
            try:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']
                )
            except Exception as e:
                print(f"[Crawler] Browser launch failed: {e}")
                raise
        return self.browser

    async def fetch_page(self, url: str) -> str:
        timeout_ms = PERFORMANCE_CONFIG["scrape_timeout"]

        try:
            browser = await self.start_browser()
            page = await browser.new_page()
            try:
                response = await page.goto(url, wait_until="networkidle", timeout=timeout_ms)

                if not response:
                    return ""

                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type.lower():
                    return ""

                html = await page.content()
                print(f"[Crawler] ✓ {url} ({len(html)} chars)")
                return html
            finally:
                await page.close()
        except asyncio.TimeoutError:
            print(f"[Crawler] ⏱ Timeout: {url}")
            return ""
        except Exception as e:
            print(f"[Crawler] ✗ Error: {url}")
            return ""

    def extract_text(self, html: str) -> str:
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "nav", "footer",
                        "header", "form", "svg", "iframe"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    async def search_relevant_pages(self, query: str, max_pages: int = None) -> str:
        max_pages = max_pages or PERFORMANCE_CONFIG["max_pages"]
        max_content = PERFORMANCE_CONFIG["max_content_per_page"]

        all_content = []

        print(f"[Web Scraper] Searching: {query}")

        to_visit = self.priority_urls.copy()

        for url in to_visit[:max_pages]:
            if url in self.visited:
                continue

            self.visited.add(url)
            html = await self.fetch_page(url)

            if html:
                text = self.extract_text(html)
                if text and len(text) > 100:
                    content_preview = text[:max_content]
                    all_content.append(f"From {url}:\n{content_preview}")

        await self.close()

        combined = "\n\n---\n\n".join(all_content)
        print(f"[Web Scraper] Total: {len(combined)} chars from {len(all_content)} pages")
        return combined

    async def close(self):
        if self.browser:
            try:
                await self.browser.close()
            except:
                pass
            self.browser = None

        if self.playwright:
            try:
                await self.playwright.stop()
            except:
                pass
            self.playwright = None


# -------------------------
# PDF Helper
# -------------------------
def extract_text_from_pdf(path: str, max_pages: int = 50) -> str:
    if PyPDF2 is None:
        return ""

    try:
        reader = PyPDF2.PdfReader(path)
        total_pages = len(reader.pages)
        pages_to_read = min(max_pages, total_pages)

        texts = []
        for i in range(pages_to_read):
            try:
                text = reader.pages[i].extract_text()
                if text:
                    texts.append(text)
            except:
                continue

        result = "\n".join(texts)
        print(f"[PDF] Extracted {pages_to_read}/{total_pages} pages from {os.path.basename(path)}")
        return result
    except Exception as e:
        print(f"[PDF] Error: {e}")
        return ""


# -------------------------
# Model Manager WITH MEMORY
# -------------------------
class ModelManager:
    """Optimized model management with conversation memory"""

    def __init__(self, loop=None):
        self.retriever = None
        self.vectorstore = None
        self.cloud_api = CloudAPIManager(loop)
        self.web_crawler = CICTWebCrawler(loop)
        self.semantic_router = SemanticRouter()
        self.loop = loop or asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=PERFORMANCE_CONFIG["max_workers"]
        )

        self.general_prompt = general_system_prompt
        self.rag_prompt = rag_system_prompt
        self.web_prompt = web_scrape_system_prompt

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": PERFORMANCE_CONFIG["retrieval_k"]}
        )
        print(f"[System] Retriever configured (k={PERFORMANCE_CONFIG['retrieval_k']})")

    async def retrieve_documents(self, question: str) -> List[Dict]:
        if not self.retriever:
            return []

        try:
            docs = await self.loop.run_in_executor(
                self.executor,
                lambda: self.retriever.invoke(question)
            )

            formatted = []
            for doc in docs:
                formatted.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", 0) + 1
                })

            print(f"[Retrieval] Found {len(formatted)} relevant chunks")
            return formatted
        except Exception as e:
            print(f"[Retrieval] Error: {e}")
            return []

    @staticmethod
    def response_has_no_info(response: str) -> bool:
        phrases = [
            "don't have information", "not mentioned", "not found",
            "couldn't find", "not available", "not in my documents",
            "no information about"
        ]
        return any(p in response.lower() for p in phrases)

    def format_conversation_history(self, chat_memory: List[Dict]) -> List[Dict]:
        """
        Format recent conversation history for API
        Returns last N turns (user + assistant pairs)
        """
        max_turns = PERFORMANCE_CONFIG["max_memory_turns"]
        max_messages = max_turns * 2  # Each turn = user + assistant

        # Get recent messages
        recent_memory = chat_memory[-max_messages:] if len(chat_memory) > max_messages else chat_memory

        return recent_memory

    async def get_response(self, question: str, chat_memory: List[Dict]) -> Tuple[str, str]:
        """
        FIXED: Now uses conversation memory!
        """

        print(f"[Query] {question}")

        # Format conversation history
        conversation_history = self.format_conversation_history(chat_memory)

        # Add current question
        current_messages = conversation_history + [{"role": "user", "content": question}]

        # STEP 1: LLM decides route
        route = await self.semantic_router.route_query(question, self.cloud_api)
        print(f"[Router] Decision: {route}")

        # ROUTE 1: GENERAL
        if route == "GENERAL":
            response = await self.cloud_api.call_with_fallbacks(
                self.general_prompt,
                current_messages,
                max_tokens=500
            )
            if response:
                return response, "General"
            return f"Hello! I'm {CICT_NAME}, how can I help?", "General (Fallback)"

        # ROUTE 2: WEB_SCRAPE
        elif route == "WEB_SCRAPE":
            print("[Router] Executing WEB_SCRAPE route...")
            web_content = await self.web_crawler.search_relevant_pages(question)

            if web_content and len(web_content) > 100:
                prompt = self.web_prompt.format(
                    cict_name=CICT_NAME,
                    query=question,
                    web_content=web_content,
                    question=question
                )
                response = await self.cloud_api.call_with_fallbacks(
                    prompt,
                    current_messages
                )
                if response:
                    return response, "Web Scraping"

            print("[Router] Web scraping insufficient, falling back to RAG...")
            route = "RAG"

        # ROUTE 3: RAG
        if route == "RAG":
            print("[Router] Executing RAG route...")
            context_docs = await self.retrieve_documents(question)

            if context_docs:
                context_text = "\n\n---\n\n".join([
                    f"Document: {doc['source']} (Page {doc['page']})\n{doc['content']}"
                    for doc in context_docs
                ])

                prompt = self.rag_prompt.format(
                    cict_name=CICT_NAME,
                    grading_context=grading_context,
                    context=context_text,
                    question=question
                )

                response = await self.cloud_api.call_with_fallbacks(
                    prompt,
                    current_messages
                )

                if response and not self.response_has_no_info(response):
                    return response, "RAG (PDF Knowledge)"

            print("[Router] RAG insufficient, trying web scraping...")
            web_content = await self.web_crawler.search_relevant_pages(question)

            if web_content and len(web_content) > 100:
                prompt = self.web_prompt.format(
                    cict_name=CICT_NAME,
                    query=question,
                    web_content=web_content,
                    question=question
                )
                response = await self.cloud_api.call_with_fallbacks(
                    prompt,
                    current_messages
                )
                if response:
                    return response, "Web Scraping (Fallback)"

        return (
            "I couldn't find information about that. Please visit bulsucict.com or try rephrasing.",
            "No Results"
        )


# -------------------------
# Response Formatter
# -------------------------
def format_response(raw_text: str) -> str:
    if not raw_text:
        return "⚠️ No response generated."

    text = raw_text.strip()
    text = re.sub(r'—\s*\*\*' + re.escape(CICT_NAME) + r'\*\*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\(.*?\.pdf.*?Page.*?\)', '', text, flags=re.IGNORECASE)

    return text.strip()


# -------------------------
# Flask App
# -------------------------
app = Flask(__name__, static_folder=None)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
model_manager: Optional[ModelManager] = None

# Chat memory - now properly used!
chat_memory: List[Dict[str, str]] = []
MAX_MEMORY_SIZE = PERFORMANCE_CONFIG["max_memory_size"]


async def init_model_manager():
    global model_manager

    if model_manager is not None:
        return

    print("[System] Initializing model manager...")
    model_manager = ModelManager(loop)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PERFORMANCE_CONFIG["chunk_size"],
        chunk_overlap=PERFORMANCE_CONFIG["chunk_overlap"],
        separators=["\n\n", "\n", ".", "!", "?", ";", ","]
    )

    faiss_path = str(FAISS_DIR)

    try:
        if os.path.exists(faiss_path) and os.path.exists(f"{faiss_path}.faiss"):
            print("[System] Loading existing FAISS index...")
            db = FAISS_local.load_local(
                faiss_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            model_manager.set_vectorstore(db)
            print("[System] ✓ FAISS loaded successfully")
        else:
            print("[System] Building FAISS index from PDFs...")
            all_texts = []

            for pdf in pdf_paths:
                if os.path.exists(pdf):
                    print(f"[System] Processing {os.path.basename(pdf)}...")
                    text = extract_text_from_pdf(pdf)

                    if text.strip():
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            all_texts.append({
                                "content": chunk,
                                "source_file": os.path.basename(pdf)
                            })

            if all_texts:
                print(f"[System] Creating FAISS with {len(all_texts)} chunks...")
                db = FAISS_local.from_texts(
                    texts=[d["content"] for d in all_texts],
                    embedding=embeddings,
                    metadatas=[{"source_file": d["source_file"]} for d in all_texts]
                )

                os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
                db.save_local(faiss_path)
                model_manager.set_vectorstore(db)
                print(f"[System] ✓ FAISS built and saved ({len(all_texts)} chunks)")
            else:
                print("[System] ⚠ No PDFs found")
    except Exception as e:
        print(f"[System] ✗ FAISS error: {e}")


@app.route("/")
def index():
    index_path = GUI_DIR / "index.html"
    if index_path.exists():
        return send_file(str(index_path))
    return f"{CICT_NAME} interface not found", 404


@app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory(str(GUI_DIR / "images"), filename)


@app.route("/<path:filepath>")
def serve_file(filepath):
    file_path = GUI_DIR / filepath
    if file_path.exists() and file_path.is_file():
        return send_from_directory(str(GUI_DIR), filepath)
    return "File not found", 404


@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """FIXED: Now properly uses conversation memory!"""
    global chat_memory

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"reply": "Invalid request", "model": "error"}), 400

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Empty message", "model": "error"}), 400

    try:
        loop.run_until_complete(init_model_manager())

        # Add user message to memory
        chat_memory.append({"role": "user", "content": message})

        # Trim memory if too large
        if len(chat_memory) > MAX_MEMORY_SIZE:
            chat_memory = chat_memory[-MAX_MEMORY_SIZE:]

        # Get response WITH conversation history
        response_raw, model_name = loop.run_until_complete(
            model_manager.get_response(message, chat_memory)
        )

        # Format response
        formatted = format_response(response_raw)

        # Add assistant response to memory
        chat_memory.append({"role": "assistant", "content": formatted})

        # Context for debugging
        context_docs = loop.run_until_complete(
            model_manager.retrieve_documents(message)
        )

        context_text = "\n\n---\n\n".join([
            f"Document: {doc['source']} (Page {doc['page']})\n{doc['content']}"
            for doc in context_docs
        ]) if context_docs else "No context retrieved"

        print(f"[Memory] Current size: {len(chat_memory)} messages")

        return jsonify({
            "reply": formatted.replace("\n", "<br>"),
            "model": model_name,
            "context": context_text
        })

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "reply": "⚠️ An error occurred. Please try again.",
            "model": "error"
        })


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "system": CICT_NAME,
        "routing": "LLM Semantic Router",
        "model_loaded": model_manager is not None,
        "memory_size": len(chat_memory),
        "config": {
            "models": GROQ_MODELS,
            "chunk_size": PERFORMANCE_CONFIG["chunk_size"],
            "retrieval_k": PERFORMANCE_CONFIG["retrieval_k"],
            "max_memory_turns": PERFORMANCE_CONFIG["max_memory_turns"]
        }
    })


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    """NEW: Endpoint to reset conversation memory"""
    global chat_memory
    chat_memory = []
    print("[Memory] Conversation reset")
    return jsonify({"status": "success", "message": "Conversation memory cleared"})


@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_loop():
        if model_manager:
            try:
                loop.run_until_complete(model_manager.cloud_api.close())
                loop.run_until_complete(model_manager.web_crawler.close())
            except:
                pass
        loop.stop()

    threading.Thread(target=stop_loop, daemon=True).start()
    return "Shutting down gracefully", 200


if __name__ == "__main__":
    print("=" * 60)
    print(f"🚀 Starting {CICT_NAME} - WITH MEMORY RETENTION")
    print("=" * 60)
    print(f"📋 Smart Features:")
    print(f"   • LLM Semantic Routing")
    print(f"   • Conversation Memory (last {PERFORMANCE_CONFIG['max_memory_turns']} turns)")
    print(f"   • Multi-model fallbacks")
    print(f"   • Connection pooling")
    print(f"   • Optimized retrieval")
    print("=" * 60)

    try:
        loop.run_until_complete(init_model_manager())
    except Exception as e:
        print(f"[WARN] Init error: {e}")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)