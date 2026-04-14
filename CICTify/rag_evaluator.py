"""
RAG Evaluation Script for CICTify Chatbot
Metrics:
- Answer Relevancy
- Faithfulness
- Context Recall
- Context Precision

Uses Groq API:
- llama-3.3-70b-versatile for main chatbot (your system)
- openai/gpt-oss-120b for evaluator scoring
"""

import asyncio
import json
import os
from typing import List, Dict, Optional
import aiohttp
from pathlib import Path
import requests
import csv

# =====================================================
# CONFIGURATION (API key removed for safety)
# =====================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EVALUATOR_MODEL = "openai/gpt-oss-120b"
MAIN_MODEL = "qwen/qwen3-32b"


# =====================================================
# TEST CASES (ALL 10 COMPLETE)
# =====================================================

TEST_CASES = [
    {
        "id": 1,
        "question": "What is the bulsu mission and vision?",
        "ground_truth": (
            "Mission: Bulacan State University exists to produce highly competent, ethical and service-oriented professionals that contribute to the sustainable socio-economic growth and development of the nation. "
            "Vision: Bulacan State University is a progressive knowledge-generating institution globally recognized for excellent instruction, pioneering research, and responsive community engagements."
        ),
        "expected_context_keywords": [
             "Bulacan State University exists to produce highly competent, ethical and service-oriented professionals that contribute to the sustainable socio-economic growth and development of the nation."
            "Bulacan State University is a progressive knowledge-generating institution globally recognized for excellent instruction, pioneering research, and responsive community engagements."
            "Mission", "Vision"

        ]
    },
    {
        "id": 2,
        "question": "What is the cict mission and vision?",
        "ground_truth": (
            "VISION: Excellence in producing globally competitive graduates in the field of Information and Communications Technology responsive to the changing needs of the society."
            "MISSION: To provide quality education by ensuring efficient and effective delivery of instruction through appropriate adoption of technological innovation and research in carrying out extension services."
        ),
        "expected_context_keywords": [
            "To provide quality education by ensuring efficient and effective delivery of instruction through appropriate adoption of technological innovation and research in carrying out extension services."
            "Excellence in producing globally competitive graduates in the field of Information and Communications Technology responsive to the changing needs of the society. "
            "College of Information and Communications Technology", "CICT", "Mission", "Vision"

        ]
    },
    {
        "id": 3,
        "question": "What are the programs offered under the College of Information and Communications Technology?",
        "ground_truth": (
            "Bachelor of Library and Information Science (BLIS), Bachelor of Science in Information System (BSIS), "
            "and Bachelor of Science in Information Technology (BSIT)"
        ),
        "expected_context_keywords": [
            "BLIS", "BSIS", "BSIT", "Library", "Library and Information Science",
            "Information System", "Information Technology", "Bachelor of Science"
        ]
    },
    {
        "id": 4,
        "question": "What are the different tracks or specializations offered in BSIT?",
        "ground_truth": "Data and Business Analytics, Infrastructure Services, Web and Mobile Applications Development.",
        "expected_context_keywords": [
            "Data and Business Analytics",
            "Infrastructure Services",
            "Web and Mobile Applications Development"
        ]
    },
    {
        "id": 5,
        "question": "Who is the dean of CICT?",
        "ground_truth": "The dean of the BulSU College of Information and Communications Technology is Dr. Digna S. Evale.",
        "expected_context_keywords": [
            "Dr. Digna S. Evale", "dean", "College of Information and Communications Technology", "BulSU"
        ]
    },
    {
        "id": 6,
        "question": "Campuses that offer Bachelor of Science in Information Technology (BSIT)?",
        "ground_truth": (
            "The Bachelor of Science in Information Technology (BSIT) program is available at Bulacan State University's "
            "Main Campus, as well as in the Bustos, Meneses, Sarmiento, and Hagonoy Campuses. However, the BSIT program "
            "is not offered at the San Rafael Campus."
        ),
        "expected_context_keywords": [
            "Main Campus", "Bustos", "Meneses", "Sarmiento", "Hagonoy", "San Rafael", "not offered"
        ]
    },
    {
        "id": 7,
        "question": "What is a shiftee?",
        "ground_truth": (
            "Shiftee is a student of BulSU who intends to change program or curriculum"
        ),
        "expected_context_keywords": [
            "student", "BulSU", "change program", "curriculum", "Shiftee", "shift", "different program or course"
        ]
    },
    {
        "id": 8,
        "question": "What is a transferee?",
        "ground_truth": (
            "Transferee is a student from another recognized institution of higher learning and is officially allowed to "
            "enroll in the same or another program in the University."
        ),
        "expected_context_keywords": [
            "student", "another institution", "higher learning", "enroll", "University", "another program", "Transferee"
        ]
    },
    {
        "id": 9,
        "question": "What is a returnee?",
        "ground_truth": (
            "Returnee is a student who was previously enrolled at the University and discontinued studies for one semester or longer for valid reasons"

        ),
        "expected_context_keywords": [
            "student", "previously enrolled", "discontinue", "semester", "reasons"
        ]
    },
    {
        "id": 10,
        "question": "What is a LOA?",
        "ground_truth": "Leave of Absence",
        "expected_context_keywords": ["Leave of Absence"]
    }
]
# =====================================================
# GROQ API CLIENT
# =====================================================

class GroqEvaluator:
    """Handles Groq API calls for evaluator model."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def call_groq(self, messages: List[Dict], model: str, max_tokens: int = 1500) -> Optional[str]:
        """Call Groq chat completion endpoint."""
        try:
            session = await self.get_session()
            url = "https://api.groq.com/openai/v1/chat/completions"

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 1.0,
                "max_tokens": max_tokens
            }

            async with session.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=aiohttp.ClientTimeout(total=45)
            ) as resp:

                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()

                print(f"[Groq API Error] HTTP {resp.status}")
                return None

        except Exception as e:
            print(f"[Groq Exception] {e}")
            return None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# =====================================================
# METRIC SCORING CLASS
# =====================================================

class RAGMetrics:
    """Computes all four RAG evaluation metrics using evaluator LLM."""

    def __init__(self, evaluator: GroqEvaluator):
        self.evaluator = evaluator

    @staticmethod
    def _clean_json(response: str) -> Dict:
        """Normalize evaluator JSON response."""
        try:
            text = response.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            return json.loads(text.strip())
        except Exception:
            return {"score": 0, "reasoning": "JSON parse error"}

    async def answer_relevancy(self, question: str, answer: str) -> Dict:
        """Score how relevant the answer is to the question."""
        prompt = f"""
Evaluate the relevancy of the answer to the question.

Question: {question}
Answer: {answer}

Rate 0-10.
Respond ONLY with JSON:
{{"score": X, "reasoning": "brief explanation"}}
"""
        resp = await self.evaluator.call_groq([{"role": "user", "content": prompt}], EVALUATOR_MODEL)

        data = self._clean_json(resp or "")
        return {"score": data.get("score", 0)/10, "reasoning": data.get("reasoning", "")}

    async def faithfulness(self, answer: str, context: str) -> Dict:
        """Score hallucination-free correctness relative to context."""
        prompt = f"""
Evaluate if the answer is faithful to the context.

Context: {context}
Answer: {answer}

Rate 0-10.
Respond ONLY with JSON:
{{"score": X, "reasoning": "brief explanation"}}
"""
        resp = await self.evaluator.call_groq([{"role": "user", "content": prompt}], EVALUATOR_MODEL)

        data = self._clean_json(resp or "")
        return {"score": data.get("score", 0)/10, "reasoning": data.get("reasoning", "")}

    async def context_recall(self, ground_truth: str, context: str) -> Dict:
        """Score whether context includes needed ground truth information."""
        prompt = f"""
Evaluate context recall vs ground truth.

Ground Truth: {ground_truth}
Retrieved Context: {context}

Rate 0-10.
Respond ONLY with JSON.
"""
        resp = await self.evaluator.call_groq([{"role": "user", "content": prompt}], EVALUATOR_MODEL)

        data = self._clean_json(resp or "")
        return {"score": data.get("score", 0)/10, "reasoning": data.get("reasoning", "")}

    async def context_precision(self, question: str, context: str, ground_truth: str) -> Dict:
        """Score ratio of relevant context vs noise."""
        prompt = f"""
Evaluate context precision for answering the question.

Question: {question}
Ground Truth Answer: {ground_truth}
Retrieved Context: {context}

Rate 0-10.
Respond ONLY with JSON.
"""
        resp = await self.evaluator.call_groq([{"role": "user", "content": prompt}], EVALUATOR_MODEL)

        data = self._clean_json(resp or "")
        return {"score": data.get("score", 0)/10, "reasoning": data.get("reasoning", "")}


# =====================================================
# CHATBOT CALL (YOUR LOCAL BACKEND)
# =====================================================

async def query_chatbot(question: str) -> Dict:
    """
    Replace this with your real `superai_web.py` backend call.
    Must return:
    {
        "answer": <chatbot output>,
        "context": <retrieved context>,
        "model": <model name>
    }
    """
    try:
        response = requests.post(
            "http://localhost:5000/chat",
            json={"message": question},
            timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            answer = data.get("reply", "").replace("<br>", "\n")

            return {
                "answer": answer,
                "context": data.get("context", "Retrieved context from FAISS"),
                "model": data.get("model", "unknown")
            }

    except Exception as e:
        print(f"[Chatbot Error] {e}")

    return {"answer": "Error: chatbot unavailable", "context": "", "model": "error"}


# =====================================================
# MAIN EVALUATION LOOP
# =====================================================

async def run_evaluation():
    print("="*60)
    print("CICTify RAG EVALUATION")
    print("="*60)

    evaluator = GroqEvaluator(GROQ_API_KEY)
    metrics = RAGMetrics(evaluator)
    results = []

    for test in TEST_CASES:
        print(f"\n[TEST {test['id']}] {test['question']}")
        print("-"*60)

        response = await query_chatbot(test["question"])
        answer = response["answer"]
        context = response["context"]

        print(f"Answer preview: {answer[:120]}...")

        relevancy = await metrics.answer_relevancy(test["question"], answer)
        faithfulness = await metrics.faithfulness(answer, context)
        recall = await metrics.context_recall(test["ground_truth"], context)
        precision = await metrics.context_precision(test["question"], context, test["ground_truth"])

        result = {
            "id": test["id"],
            "question": test["question"],
            "ground_truth": test["ground_truth"],
            "expected_context_keywords": ", ".join(test["expected_context_keywords"]),
            "answer": answer,
            "relevancy": relevancy["score"],
            "faithfulness": faithfulness["score"],
            "recall": recall["score"],
            "precision": precision["score"]
        }
        results.append(result)

        print(f"✓ Relevancy:   {relevancy['score']:.2f}")
        print(f"✓ Faithfulness:{faithfulness['score']:.2f}")
        print(f"✓ Recall:      {recall['score']:.2f}")
        print(f"✓ Precision:   {precision['score']:.2f}")

    await evaluator.close()

    # Save results to CSV
    csv_file = Path("rag_evaluation_results_qwen.csv")
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "question", "ground_truth", "expected_context_keywords",
                "answer", "relevancy", "faithfulness", "recall", "precision"
            ],
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Aggregate scores
    avg = lambda k: sum(r[k] for r in results) / len(results)
    print("\n" + "="*60)
    print("AGGREGATE SCORES")
    print("="*60)
    print(f"Answer Relevancy: {avg('relevancy'):.2f}")
    print(f"Faithfulness:     {avg('faithfulness'):.2f}")
    print(f"Context Recall:   {avg('recall'):.2f}")
    print(f"Context Precision:{avg('precision'):.2f}")
    print("\nEvaluation Completed.")
    print(f"Results saved to: {csv_file}")

# Run if executed directly
if __name__ == "__main__":
    asyncio.run(run_evaluation())