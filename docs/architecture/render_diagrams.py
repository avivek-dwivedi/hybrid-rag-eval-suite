"""Render Mermaid diagrams to PNG using mermaid.ink and save them to disk.

The mermaid.ink URL format is:
    https://mermaid.ink/img/pako:<base64-url-encoded-pako-deflated-source>?type=png

The "pako" encoding is the same one produced by the Mermaid Live Editor's
URL hash: zlib-compress (raw, wbits=-15) then base64-url-encode (no padding).
"""

from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
TIMEOUT = 60

DIAGRAMS: dict[str, str] = {
    "pipeline.png": """flowchart LR
    C[Corpus] --> CH[Chunker]
    CH --> E[Embedder]
    E --> V[FAISS Index]
    CH --> T[BM25 Index]
    Q[Query] --> DR[Dense Retriever]
    Q --> SR[Sparse Retriever]
    V --> DR
    T --> SR
    DR --> RRF[RRF Fusion]
    SR --> RRF
    RRF --> HC[Hybrid Top-k]
    DR --> DC[Dense-only Top-k]
    HC --> HA[Hybrid Answer]
    DC --> DA[Dense-only Answer]
    HA --> EV[Eval Harness]
    DA --> EV
""",
    "framework.png": """flowchart TD
    A[Synthetic Corpus] --> B[Chunking]
    B --> C[BM25 Index]
    B --> D[Dense Index]
    C --> E[Retrieval Layer]
    D --> E
    E --> F[Top-k Contexts]
    F --> G[Generation Layer]
    G --> H[Answers]
    F --> I[Retrieval Metrics]
    H --> J[Generation Metrics]
    I --> K[Scorecard]
    J --> K
""",
}


def pako_encode(source: str) -> str:
    """Pako-deflate + base64-url-encode (matches Mermaid Live Editor)."""
    raw = zlib.compress(source.encode("utf-8"))[2:-4]  # strip zlib header + adler32
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def render(filename: str, source: str) -> Path:
    encoded = pako_encode(source)
    url = f"https://mermaid.ink/img/pako:{encoded}?type=png&bgColor=!white&theme=neutral"
    print(f"[render] {filename} <- {url[:80]}...")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/png,image/*;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = resp.read()
    if len(data) < 200:
        raise RuntimeError(f"Suspiciously small response ({len(data)} bytes) for {filename}")
    out_path = OUT_DIR / filename
    out_path.write_bytes(data)
    print(f"[render] wrote {out_path} ({len(data)} bytes)")
    return out_path


def main() -> None:
    results = []
    for name, src in DIAGRAMS.items():
        try:
            results.append({"file": name, "status": "ok", "path": str(render(name, src))})
        except Exception as exc:  # noqa: BLE001
            results.append({"file": name, "status": "error", "error": str(exc)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
