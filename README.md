# Hybrid RAG Eval Suite

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/jupyter-7.x-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![LangChain](https://img.shields.io/badge/langchain-1.2.18-1C3C3C)](https://python.langchain.com/)
[![Ragas](https://img.shields.io/badge/ragas-0.4.3-0F766E)](https://docs.ragas.io/)
[![Repo](https://img.shields.io/badge/github-avivek--dwivedi-181717?logo=github)](https://github.com/avivek-dwivedi/hybrid-rag-eval-suite)
[![Stars](https://img.shields.io/github/stars/avivek-dwivedi/hybrid-rag-eval-suite?style=social)](https://github.com/avivek-dwivedi/hybrid-rag-eval-suite/stargazers)

> A reproducible evaluation harness for **Hybrid Retrieval-Augmented Generation** pipelines. Compares dense retrieval, BM25 sparse retrieval, and their Reciprocal Rank Fusion (RRF) hybrid — measured with retrieval metrics, generation metrics, Ragas, and an optional LLM-as-judge.

> **Status:** active development. The evaluation harness is working end-to-end and is being prepared for a public v1.0 release. Expect API surface and notebook structure to evolve until the v1.0 tag is cut.

---

## Why this exists

Production RAG systems fail in two distinct places: **retrieval** (wrong chunks) and **generation** (right chunks, wrong answer). Most public RAG demos measure only the latter. This suite isolates both layers and reports them separately, so you can debug a regression without guessing where it came from.

The pipeline follows a strict layering principle:

```mermaid
flowchart LR
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
```

---

## Framework overview

```mermaid
flowchart TD
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
```

Each layer is independently measurable. A drop in the final scorecard points to exactly one layer.

---

## Repository layout

```
hybrid-rag-eval-suite/
|-- hybrid_rag_demo.ipynb          # End-to-end demo: dense vs hybrid vs LLM judge
|-- hybrid_rag_evaluation.ipynb    # Full evaluation harness with scorecards
|-- hybrid_rag_theory.ipynb        # Reference notes: BM25, RRF, BLEU, NDCG, MRR
|-- synthetic_rag_dataset.csv      # Flat synthetic corpus + queries
|-- synthetic_rag_dataset.jsonl    # JSONL view, same data
|-- requirements.txt               # Pinned pip dependencies
|-- environment.yml                # Conda mirror of requirements.txt
|-- .env.example                   # Template for optional Groq config
`-- README.md
```

---

## What is implemented

### 1. Retrieval layer

| Retriever | Algorithm | Notes |
| --- | --- | --- |
| Dense | FAISS over sentence embeddings | Strong on semantic / paraphrase match |
| Sparse | BM25 (rank-bm25) | Strong on exact-token / keyword match |
| Hybrid | Reciprocal Rank Fusion over both | Parameter-free rank fusion |

### 2. Generation layer

- **Extractive fallback** — runs without any external LLM
- **Groq-backed generation** — optional, via `langchain-groq`

### 3. Evaluation layer

**Retrieval metrics**

- Source precision@k, source recall@k
- Chunk precision@k, chunk recall@k
- Top-1 source correctness, first relevant rank
- MRR, NDCG@k
- Ragas context precision, Ragas context recall

**Generation metrics**

- Manual exact match
- Required-fact recall, all-required-facts-present, forbidden-fact hits
- Answer-context overlap
- Ragas exact match, Ragas BLEU, Ragas ROUGE
- LLM judge: groundedness, alignment, completeness

---

## Installation

### Option A — venv (recommended for development)

```bash
# clone
git clone https://github.com/avivek-dwivedi/hybrid-rag-eval-suite.git
cd hybrid-rag-eval-suite

# create + activate
python -m venv .venv
source .venv/bin/activate           # macOS / Linux
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# install
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name hybrid-rag-eval \
        --display-name "Python (hybrid-rag-eval)"
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate hybrid-rag-eval
python -m ipykernel install --user --name hybrid-rag-eval \
        --display-name "Python (hybrid-rag-eval)"
```

Refresh an existing environment:

```bash
conda env update -f environment.yml --prune
```

---

## Configuration

The notebooks run end-to-end with **zero external services**. The Groq integration is only enabled if you provide credentials.

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | no | Enables Groq-based generation and LLM judge |
| `GROQ_MODEL` | no | Overrides the default Groq model |

---

## Usage

Run the notebooks in this order:

1. `hybrid_rag_demo.ipynb` — end-to-end pipeline, side-by-side dense vs hybrid answers, LLM judge
2. `hybrid_rag_evaluation.ipynb` — full retrieval + generation metric suite with scorecards
3. `hybrid_rag_theory.ipynb` — reference notes for every metric and algorithm used

Start Jupyter:

```bash
jupyter notebook
```

---

## Reproducibility

- The corpus and queries are synthetic and version-controlled, so retrieval runs are deterministic across machines.
- Groq-backed generation is non-deterministic by nature and varies by model and time.
- Without `GROQ_API_KEY`, the pipeline falls back to extractive answers and the full metric suite still runs.
- NLTK tokenizers are downloaded on first run only.

---

## Pinned dependencies

| Package | Version | Role |
| --- | --- | --- |
| langchain | 1.2.18 | Orchestration |
| langchain-core | 1.3.3 | Core abstractions |
| langchain-community | 0.4.1 | Community integrations |
| langchain-text-splitters | 1.1.1 | Chunking |
| langchain-groq | 1.1.2 | Optional Groq LLM backend |
| ragas | 0.4.3 | Ragas metric suite |
| rank-bm25 | 0.2.2 | BM25 sparse retriever |
| sacrebleu | 2.5.1 | BLEU generation metric |
| rouge-score | 0.1.2 | ROUGE generation metric |
| nltk | 3.9.3 | Tokenization |
| pandas | 3.0.1 | Data handling |
| pydantic | 2.11.10 | Schema validation |
| python-dotenv | 1.2.2 | `.env` loading |

See [requirements.txt](requirements.txt) for the canonical list and [environment.yml](environment.yml) for the conda mirror.

---

## Roadmap

The roadmap below is structured as a publish-readiness plan. The first column is **in scope for v1.0**; the second column is what lands **after** the first public release.

### v1.0 — Publish

- [ ] Finalize the public API for the retrieval layer (`dense_retrieve`, `bm25_retrieve`, `rrf_fuse`)
- [ ] Lock notebook structure: `demo` + `evaluation` + `theory`, with no breaking changes between minor versions
- [ ] Add a `LICENSE` file (MIT) before tagging v1.0
- [ ] Write a `CONTRIBUTING.md` and an issue / PR template set
- [ ] Add GitHub Actions CI that runs the synthetic evaluation on every push and PR
- [ ] Cross-encoder reranking experiment behind a feature flag
- [ ] Dense retriever baseline numbers reported alongside BM25 in the scorecard
- [ ] CSV / Markdown export of scorecards for downstream reporting
- [ ] Side-by-side pipeline comparison cells in the demo notebook
- [ ] Cut the **v1.0** release tag and publish a PyPI / GitHub release with packaged notebooks

### Post-v1.0 — Iterate

- [ ] Containerized notebook environment (Docker / devcontainer)
- [ ] Pluggable retriever interface so users can drop in their own encoder / sparse index
- [ ] Multi-corpus evaluation harness (load and benchmark multiple datasets in one run)
- [ ] Web UI for browsing scorecards across runs
- [ ] Optional reranker presets (cross-encoder, ColBERT, LLM-based)

> Want to influence what lands in v1.0? Open an issue or start a discussion on the repository.

---

## License

Released without a license file — all rights reserved by the author. Add a `LICENSE` file before publishing publicly if you intend to grant reuse rights.

---

## Author

**Avivek Dwivedi** — AI Engineer working on retrieval systems, evaluation harnesses, and applied LLM tooling.

- GitHub: [@avivek-dwivedi](https://github.com/avivek-dwivedi)
- Repository: [hybrid-rag-eval-suite](https://github.com/avivek-dwivedi/hybrid-rag-eval-suite)