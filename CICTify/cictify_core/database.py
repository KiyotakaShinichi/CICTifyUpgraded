import os
import sqlite3
from typing import List, Tuple

from .config import DB_PATH


SEED_FACTS: List[Tuple[str, str, str]] = [
    ("dean cict", "The dean of BulSU CICT is Dr. Digna S. Evale.", "faq"),
    ("programs offered", "BulSU CICT offers BLIS, BSIS, and BSIT.", "faq"),
    ("what is loa", "LOA means Leave of Absence.", "faq"),
    ("what is shiftee", "A shiftee is a BulSU student intending to change program or curriculum.", "faq"),
]


class StructuredDB:
    def __init__(self) -> None:
        self.path = str(DB_PATH)

    def initialize(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT UNIQUE,
                    answer TEXT NOT NULL,
                    source TEXT DEFAULT 'seed'
                )
                """
            )
            for key, answer, source in SEED_FACTS:
                conn.execute(
                    "INSERT OR IGNORE INTO facts(key, answer, source) VALUES (?, ?, ?)",
                    (key, answer, source),
                )
            conn.commit()

    def lookup(self, question: str, limit: int = 3) -> List[Tuple[str, str, str]]:
        text = (question or "").lower().strip()
        if not text:
            return []

        terms = [t for t in text.split() if len(t) > 2][:6]
        if not terms:
            return []

        where = " OR ".join(["LOWER(key) LIKE ?"] * len(terms))
        params = [f"%{term}%" for term in terms]

        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                f"SELECT key, answer, source FROM facts WHERE {where} LIMIT ?",
                (*params, limit),
            ).fetchall()
        return rows

    def context_for(self, question: str) -> str:
        rows = self.lookup(question)
        if not rows:
            return ""
        return "\n".join([f"- {key}: {answer} (source={source})" for key, answer, source in rows])
