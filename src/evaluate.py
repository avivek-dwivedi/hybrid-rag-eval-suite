"""Evaluate BM25, dense, hybrid-rrf, and optional reranked hybrid retrieval."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.data import load_documents, load_eval_queries
from src.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    DenseIndex,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rrf_fuse,
    tokenize,
    write_json,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DEFAULT_K = 5
DEFAULT_RRF_K = 60
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FETCH_K = 10
METRICS = ("recall@k", "mrr", "ndcg@k", "precision@k", "hit@k")


def rank_strategies(
    query: str,
    documents: list[dict],
    doc_ids: list[str],
    bm25_index,
    dense_index,
    top_k: int,
    rrf_k: int,
    reranker: CrossEncoderReranker | None,
) -> dict[str, list[str]]:
    """Rank doc ids per strategy for one query."""
    bm25_ranked = bm25_index.search(query, FETCH_K)
    dense_ranked = dense_index.search(query, FETCH_K)
    fused = rrf_fuse([bm25_ranked, dense_ranked], k=rrf_k)[:top_k]

    strategies = {
        "bm25": [doc_ids[pos] for pos, _ in bm25_ranked[:top_k]],
        "dense": [doc_ids[pos] for pos, _ in dense_ranked[:top_k]],
        "hybrid-rrf": [doc_ids[pos] for pos, _ in fused],
    }

    if reranker is not None:
        fused_ids = [doc_ids[pos] for pos, _ in fused]
        reranked = reranker.rerank(
            query, fused_ids, {doc["id"]: doc["text"] for doc in documents},
        )
        strategies["hybrid-rrf-rerank"] = [doc_id for doc_id, _score in reranked]

    return strategies


def evaluate(k: int, embedding_model: str, cross_encoder_model: str, skip_rerank: bool) -> dict:
    documents = load_documents()
    queries = load_eval_queries()
    doc_ids = [doc["id"] for doc in documents]

    bm25_index = BM25Index([tokenize(doc["text"]) for doc in documents])
    dense_index = DenseIndex(embedding_model).load().build([doc["text"] for doc in documents])
    reranker = None if skip_rerank else CrossEncoderReranker(cross_encoder_model)

    strategy_names = ["bm25", "dense", "hybrid-rrf"] + ([] if skip_rerank else ["hybrid-rrf-rerank"])
    scores: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in METRICS} for name in strategy_names
    }
    per_query: list[dict] = []

    for query in queries:
        strategies = rank_strategies(
            query["query"], documents, doc_ids, bm25_index, dense_index,
            top_k=k, rrf_k=DEFAULT_RRF_K, reranker=reranker,
        )
        gold_ids = set(query["relevant_ids"])

        for strategy in strategy_names:
            retrieved_ids = strategies[strategy]
            per_query.append({
                "query_id": query["id"],
                "query": query["query"],
                "strategy": strategy,
                "results": [
                    {"rank": rank, "doc_id": doc_id,
                     "title": next(d["title"] for d in documents if d["id"] == doc_id)}
                    for rank, doc_id in enumerate(retrieved_ids, start=1)
                ],
                "relevant_ids": query["relevant_ids"],
            })
            scores[strategy]["recall@k"].append(recall_at_k(retrieved_ids, gold_ids, k))
            scores[strategy]["mrr"].append(reciprocal_rank(retrieved_ids, gold_ids))
            scores[strategy]["ndcg@k"].append(ndcg_at_k(retrieved_ids, gold_ids, k))
            scores[strategy]["precision@k"].append(precision_at_k(retrieved_ids, gold_ids, k))
            scores[strategy]["hit@k"].append(hit_at_k(retrieved_ids, gold_ids, k))

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    metrics = [
        {
            "strategy": name,
            f"recall@{k}": mean(scores[name]["recall@k"]),
            "mrr": mean(scores[name]["mrr"]),
            f"ndcg@{k}": mean(scores[name]["ndcg@k"]),
            f"precision@{k}": mean(scores[name]["precision@k"]),
            f"hit@{k}": mean(scores[name]["hit@k"]),
        }
        for name in strategy_names
    ]

    evaluation = {
        "generated_by": "python -m src.evaluate",
        "k": k,
        "rrf_k": DEFAULT_RRF_K,
        "embedding_model": embedding_model,
        "cross_encoder_model": None if skip_rerank else cross_encoder_model,
        "corpus_size": len(documents),
        "query_count": len(queries),
        "metrics": metrics,
        "per_query": per_query,
    }

    write_json(RESULTS_DIR / "evaluation.json", evaluation)
    write_json(RESULTS_DIR / "retrieval-results.json", per_query)
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate hybrid RAG retrieval strategies.")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--cross-encoder-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()

    evaluation = evaluate(
        k=args.k,
        embedding_model=args.embedding_model,
        cross_encoder_model=args.cross_encoder_model,
        skip_rerank=args.skip_rerank,
    )

    print("Evaluation complete.")
    print(f"  corpus={evaluation['corpus_size']} queries={evaluation['query_count']} k={evaluation['k']} rrf_k={evaluation['rrf_k']}")
    print(f"  embedding model: {evaluation['embedding_model']}")
    print(f"  cross-encoder:   {evaluation['cross_encoder_model'] or 'not run (--skip-rerank)'}")
    for row in evaluation["metrics"]:
        cells = "  ".join(f"{key}={value}" for key, value in row.items() if key != "strategy")
        print(f"  {row['strategy']:<20} {cells}")
    print(f"  artifacts: results/evaluation.json, results/retrieval-results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())