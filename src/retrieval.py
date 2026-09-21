"""Tokenization, BM25, dense retrieval, RRF fusion, and retrieval metrics."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
DEFAULT_RRF_K = 60


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_tokens)
        self.doc_lens = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.doc_count if self.doc_count else 0.0

        self.doc_term_freqs = [Counter(tokens) for tokens in corpus_tokens]
        self.term_doc_freq: Counter[str] = Counter()
        for tf in self.doc_term_freqs:
            self.term_doc_freq.update(tf.keys())

        self.inverted: dict[str, list[int]] = {}
        for doc_id, tf in enumerate(self.doc_term_freqs):
            for term in tf:
                self.inverted.setdefault(term, []).append(doc_id)

        # BM25 IDF variant with +1.0 inside the log keeps scores non-negative
        # when a term appears in more than half the corpus.
        self._doc_idf: list[dict[str, float]] = [
            {
                term: math.log(
                    (self.doc_count - self.term_doc_freq[term] + 0.5)
                    / (self.term_doc_freq[term] + 0.5)
                    + 1.0
                )
                for term in tf
            }
            for tf in self.doc_term_freqs
        ]

    def score(self, query_tokens: list[str], doc_id: int) -> float:
        tf_map = self.doc_term_freqs[doc_id]
        doc_len = self.doc_lens[doc_id]
        idf_map = self._doc_idf[doc_id]
        total = 0.0
        for term in query_tokens:
            tf = tf_map.get(term)
            if tf is None:
                continue
            idf = idf_map.get(term)
            if idf is None:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            total += idf * (tf * (self.k1 + 1)) / denom
        return total

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.doc_count:
            return []
        candidates = {
            doc_id
            for token in query_tokens
            for doc_id in self.inverted.get(token, ())
        }
        scored = [(doc_id, self.score(query_tokens, doc_id)) for doc_id in candidates]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


class DenseIndex:
    """Sentence embeddings with cosine similarity; CPU by default.

    Uses sentence-transformers when available and falls back to a direct
    transformers implementation so no extra dependency is required.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._encoder = None
        self._encode_fn = None
        self._matrix = None

    def load(self) -> "DenseIndex":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device=self.device)
            self._encoder = model
            self._encode_fn = None
        except Exception as exc:
            print(f"[dense] sentence-transformers unavailable ({exc}); using transformers fallback.")
            self._load_transformers_fallback()
        return self

    def _load_transformers_fallback(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        def encode(texts: list[str]) -> list[list[float]]:
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=256, return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            return pooled.cpu().tolist()

        self._encoder = None
        self._encode_fn = encode

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is not None:
            vectors = self._encoder.encode(
                texts, batch_size=16, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True,
            )
            return [vector.tolist() for vector in vectors]
        if self._encode_fn is None:
            raise RuntimeError("DenseIndex.load() must be called before encode().")
        return self._encode_fn(texts)

    def build(self, corpus_texts: list[str]) -> "DenseIndex":
        self._matrix = self.encode(corpus_texts)
        return self

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self._matrix is None:
            raise RuntimeError("DenseIndex.build() must be called before search().")
        query_vector = self.encode([query])[0]
        scored = [
            (doc_id, sum(q * d for q, d in zip(query_vector, doc_vector)))
            for doc_id, doc_vector in enumerate(self._matrix)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker loaded once and reused across queries."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self.failed = False

    def load(self) -> "CrossEncoderReranker":
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self.model_name, max_length=self.max_length, device=self.device,
        )
        return self

    def rerank(self, query: str, doc_ids: list[str], texts: dict[str, str]) -> list[tuple[str, float]]:
        if self.failed:
            return [(doc_id, 0.0) for doc_id in doc_ids]
        if self._model is None:
            try:
                self.load()
            except Exception as exc:
                print(f"[rerank] unavailable ({exc}); keeping fused order.")
                self.failed = True
                return [(doc_id, 0.0) for doc_id in doc_ids]
        try:
            pairs = [[query, texts[doc_id]] for doc_id in doc_ids]
            scores = self._model.predict(pairs)
            ranked = sorted(
                zip(doc_ids, (float(s) for s in scores)),
                key=lambda pair: (-pair[1], pair[0]),
            )
            return ranked
        except Exception as exc:
            print(f"[rerank] failed ({exc}); keeping fused order.")
            self.failed = True
            return [(doc_id, 0.0) for doc_id in doc_ids]


def rrf_fuse(
    ranked_lists: Iterable[list[tuple[int, float]]],
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Fuse ranked lists by summing 1 / (k + rank) per document.

    Each list is [(doc_id, score), ...] in rank order, rank starting at 1.
    Ties are broken by doc id for deterministic output.
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & gold_ids) / len(gold_ids)


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for doc_id in top if doc_id in gold_ids) / len(top)


def reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not gold_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in gold_ids
    )
    ideal_hits = min(len(gold_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def hit_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    return float(any(doc_id in gold_ids for doc_id in retrieved_ids[:k]))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")