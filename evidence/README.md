# Evidence

Screenshots in this folder were captured manually on a local Windows machine while running the commands documented in the repository README. They are the human-visible proof of the local runs; the machine-readable artifacts in `results/` are the reproducible evidence.

| File | What it shows | What it proves |
| --- | --- | --- |
| `01-local-query.png` | Terminal output of `python -m src.query "What is hybrid retrieval?"` with ranked results | The project executed locally and returned ranked hybrid retrieval results |
| `02-evaluation.png` | Terminal output of `python -m src.evaluate` with per-strategy metrics | The evaluation pipeline executed against the local evaluation set and generated real retrieval metrics |
| `03-results-report.png` | The local experiment report page (evaluation table + example query) | The generated experiment results are viewable as a static local browser report |
| `04-architecture.png` | `docs/architecture.svg` rendered in a browser | The retrieval architecture documented by the repository |

Note: `03-results-report.png` shows the report page content but not the browser address bar; the report is served locally with `python -m http.server 8000` at `http://localhost:8000/results/report.html`.

Screenshots are only added to this folder after the corresponding command actually ran; nothing here is generated or staged automatically.