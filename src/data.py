"""Load the bundled sample corpus and labeled evaluation queries."""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
EVAL_PATH = DATA_DIR / "eval-queries.jsonl"


def load_documents(path: Path | None = None) -> list[dict]:
    path = path or DOCUMENTS_PATH
    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents


def load_eval_queries(path: Path | None = None) -> list[dict]:
    path = path or EVAL_PATH
    queries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries