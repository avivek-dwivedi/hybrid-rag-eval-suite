"""Run a single hybrid retrieval query and print ranked results."""

from __future__ import annotations

import argparse
import os
import sys

from src.data import load_documents
from src.evaluate import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RRF_K,
    rank_strategies,
)
from src.retrieval import BM25Index, CrossEncoderReranker, DenseIndex, tokenize

SEPARATOR = "=" * 70


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a hybrid RAG retrieval query.")
    parser.add_argument("query", help="Query text to retrieve for.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to show.")
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--rerank", action="store_true", help="Enable optional cross-encoder reranking.")
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    args = parser.parse_args()

    documents = load_documents()
    doc_ids = [doc["id"] for doc in documents]
    bm25_index = BM25Index([tokenize(doc["text"]) for doc in documents])
    dense_index = DenseIndex(args.embedding_model).load().build([doc["text"] for doc in documents])
    reranker = CrossEncoderReranker(args.cross_encoder_model) if args.rerank else None

    strategies = rank_strategies(
        args.query, documents, doc_ids, bm25_index, dense_index,
        top_k=args.k, rrf_k=args.rrf_k, reranker=reranker,
    )
    ranked = strategies["hybrid-rrf-rerank" if args.rerank else "hybrid-rrf"]

    print()
    print(SEPARATOR)
    print("QUERY")
    print(SEPARATOR)
    print(args.query)
    print()
    print("Strategy:")
    print("hybrid-rrf-rerank" if args.rerank else "hybrid-rrf")
    print()
    print("Top results:")
    print("-" * 70)
    for rank, doc_id in enumerate(ranked, start=1):
        doc = next(d for d in documents if d["id"] == doc_id)
        print(f"{rank:02d}  {doc_id}")
        print(f"    title: {doc['title']}")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    sys.exit(main())