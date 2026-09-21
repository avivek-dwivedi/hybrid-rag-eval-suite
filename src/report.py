"""Render results/report.html from results/evaluation.json via a template."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TEMPLATE_PATH = Path(__file__).resolve().parent / "report_template.html"

PLACEHOLDERS = [
    "CORPUS_SIZE", "QUERY_COUNT", "EMBEDDING_MODEL", "STRATEGIES", "K", "RRF_K",
    "CROSS_ENCODER", "METRIC_HEADERS", "METRIC_ROWS",
    "EXAMPLE_QUERY", "EXAMPLE_STRATEGY", "EXAMPLE_RELEVANT", "EXAMPLE_ROWS",
]


def render_metric_rows(metrics: list[dict]) -> str:
    rows = []
    for row in metrics:
        keys = [key for key in row if key != "strategy"]
        cells = "".join(
            f"<td class=\"num\">{row[key]:.4f}</td>" for key in keys
        )
        rows.append(f"<tr><td><strong>{html.escape(row['strategy'])}</strong></td>{cells}</tr>")
    return "".join(rows)


def render_example_rows(example: dict) -> str:
    relevant = set(example.get("relevant_ids", []))
    rows = []
    for result in example.get("results", []):
        hit = " class=\"hit\">✔" if result["doc_id"] in relevant else ">"
        rows.append(
            f"<tr><td>{result['rank']}</td>"
            f"<td><code>{html.escape(result['doc_id'])}</code></td>"
            f"<td>{html.escape(result['title'])}</td>"
            f"<td{hit}</td></tr>"
        )
    return "".join(rows)


def build_report(evaluation: dict) -> str:
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    metrics = evaluation.get("metrics", [])
    metric_keys = [key for key in metrics[0] if key != "strategy"] if metrics else []
    example = next(
        (entry for entry in evaluation.get("per_query", []) if entry.get("strategy") == "hybrid-rrf"),
        None,
    )
    if example is None and evaluation.get("per_query"):
        example = evaluation["per_query"][0]

    values = {
        "CORPUS_SIZE": str(evaluation.get("corpus_size", "?")),
        "QUERY_COUNT": str(evaluation.get("query_count", "?")),
        "EMBEDDING_MODEL": html.escape(str(evaluation.get("embedding_model", "n/a"))),
        "STRATEGIES": html.escape(", ".join(row["strategy"] for row in metrics)),
        "K": str(evaluation.get("k", "?")),
        "RRF_K": str(evaluation.get("rrf_k", "?")),
        "CROSS_ENCODER": html.escape(str(evaluation.get("cross_encoder_model") or "not run")),
        "METRIC_HEADERS": "".join(f"<th class=\"num\">{html.escape(key)}</th>" for key in metric_keys),
        "METRIC_ROWS": render_metric_rows(metrics),
        "EXAMPLE_QUERY": html.escape(example["query"]) if example else "n/a",
        "EXAMPLE_STRATEGY": html.escape(example["strategy"]) if example else "n/a",
        "EXAMPLE_RELEVANT": " ".join(
            f"<code>{html.escape(doc_id)}</code>" for doc_id in example.get("relevant_ids", [])
        ) if example else "n/a",
        "EXAMPLE_ROWS": render_example_rows(example) if example else "",
    }
    for name in PLACEHOLDERS:
        template = template.replace(f"%%{name}%%", values[name])
    return template


def main() -> int:
    evaluation_path = RESULTS_DIR / "evaluation.json"
    if not evaluation_path.exists():
        print("evaluation.json not found. Run `python -m src.evaluate` first.")
        return 1

    with evaluation_path.open("r", encoding="utf-8") as handle:
        evaluation = json.load(handle)

    report_path = RESULTS_DIR / "report.html"
    report_path.write_text(build_report(evaluation), encoding="utf-8")
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())