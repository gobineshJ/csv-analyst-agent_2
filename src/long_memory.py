"""Long-term memory and retrieval (RAG): context that outlives one conversation.

memory.py remembers within a session. This module adds two things the model can't know
from the current conversation:

1. LongTermMemory: a SQLite file of past questions/answers and facts the user asked to
   remember, kept across program runs. Notes are tied to a fingerprint of the CSV's
   contents, so if the file changes, old answers about it are not used (they'd be stale).
2. A data dictionary (data/data_dictionary.md): business definitions that are not in the
   data itself ("a big order means sales of 900 or more"). It is split into chunks and only
   the chunks relevant to the question are added to the prompt: retrieval-augmented generation.

Retrieval here is BM25 keyword scoring in plain Python. Real systems usually use embeddings
(a vector per chunk, nearest neighbours by similarity), which also match synonyms; BM25 is
used here so there's no extra model or dependency, and the ranking is easy to inspect.

Risk: memory poisoning. A wrong answer that gets stored will be "remembered" and repeated.
Use store_if to keep only answers that passed a check (e.g. the reflection reviewer).
"""
import hashlib
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

from src.agent import SYSTEM, run_agent

STOPWORDS = set("a an and are as at be by for from has have how i in is it its me my of on or our "
                "please show tell than that the their this to was were what when where which who "
                "why will with you your".split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def bm25_rank(query, docs, k=3, k1=1.5, b=0.75):
    """Return indexes of the k docs that best match the query (only docs with a positive score)."""
    doc_tokens = [tokens(d) for d in docs]
    if not doc_tokens:
        return []
    avg_len = sum(map(len, doc_tokens)) / len(doc_tokens) or 1
    df = Counter(w for t in doc_tokens for w in set(t))
    n = len(docs)
    scores = []
    for i, t in enumerate(doc_tokens):
        tf = Counter(t)
        score = 0.0
        for w in set(tokens(query)):
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                score += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg_len))
        scores.append((score, i))
    return [i for s, i in sorted(scores, reverse=True)[:k] if s > 0]


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def load_chunks(path):
    """Split a markdown document into one chunk per '## ' section."""
    text = Path(path).read_text(encoding="utf-8")
    return [c.strip() for c in re.split(r"\n(?=## )", text) if c.strip().startswith("## ")]


class LongTermMemory:
    def __init__(self, db_path="memory.db"):
        self.db = sqlite3.connect(db_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, dataset TEXT, "
                        "kind TEXT, text TEXT, created REAL)")

    def add(self, dataset, kind, text):
        self.db.execute("INSERT INTO notes (dataset, kind, text, created) VALUES (?, ?, ?, ?)",
                        (dataset, kind, text, time.time()))
        self.db.commit()

    def remember(self, dataset, fact):
        self.add(dataset, "fact", fact)

    def notes(self, dataset):
        return [r[0] for r in self.db.execute(
            "SELECT text FROM notes WHERE dataset = ? ORDER BY created", (dataset,))]

    def recall(self, dataset, query, k=3, recent=2):
        """The k best keyword matches plus the most recent notes ("my last question" has no keywords)."""
        notes = self.notes(dataset)
        picked = set(bm25_rank(query, notes, k)) | set(range(max(0, len(notes) - recent), len(notes)))
        return [notes[i] for i in sorted(picked)]


def build_system(question, store=None, dataset=None, chunks=None, verbose=True):
    """The normal system prompt plus whatever is retrieved for this question."""
    parts = [SYSTEM]
    if chunks:
        found = [chunks[i] for i in bm25_rank(question, chunks, k=2)]
        if found:
            parts.append("Definitions from the data dictionary (use them when they apply):\n" + "\n\n".join(found))
            if verbose:
                print("RETRIEVE " + " | ".join(c.splitlines()[0] for c in found))
    if store and dataset:
        notes = store.recall(dataset, question)
        if notes:
            parts.append("Notes from earlier sessions about this same file (check numbers with tools "
                         "if they matter):\n" + "\n".join(f"- {n}" for n in notes))
            if verbose:
                print(f"RECALL   {len(notes)} note(s) from earlier sessions")
    return "\n\n".join(parts)


def run_agent_with_long_memory(question, tools, store=None, chunks=None, verbose=True, store_if=None, **kwargs):
    """run_agent with retrieved context, then save the Q&A for future sessions."""
    dataset = fingerprint(tools.path)
    answer = run_agent(question, tools, verbose=verbose,
                       system=build_system(question, store, dataset, chunks, verbose), **kwargs)
    keep = not answer.startswith("Stopped") and (store_if is None or store_if(question, answer))
    if store and keep:
        store.add(dataset, "qa", f"Q: {question} A: {answer[:500]}")
    return answer
