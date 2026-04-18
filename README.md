# Energetic Molecular Design System (EMDS)

A beam-search-based molecular designer for energetic materials. Targets four properties simultaneously (density, detonation velocity, detonation pressure, solid-phase heat of formation), uses a curated pool of 81 SMARTS-driven modification strategies, and augments XGBoost property prediction with retrieval from the scientific literature (OpenAlex, ArXiv, Crossref, Semantic Scholar).

## Features

- **Beam-search optimization** over a principled strategy pool (3⁴ = 81 direction tuples mapped by L1-nearest Δ to a validated library of ~35 SMARTS primitives).
- **Property prediction** via XGBoost models for Density, Det. Velocity, Det. Pressure, and Hf solid.
- **Feasibility filter** using SAScore.
- **Retrieval-Augmented Generation (RAG)**: full-text ArXiv PDFs + abstract databases, with embedding-based chunk retrieval, regex+LLM extraction, and a persistent SQLite cache.
- **GUI**: clean, light Flask web interface with Server-Sent Event progress streaming.

## Architecture

```
EnergeticGraph/
├── config.py                  # Configuration dataclasses
├── data_structures.py         # MoleculeState, PropertyTarget
├── descriptors.py             # Molecular descriptor generation
├── designer.py                # High-level EnergeticDesigner entry
├── main.py                    # CLI entry point
├── orchestrator.py            # BeamSearchEngine with observer callbacks
├── agents/worker_agent.py     # ChemistAgent (accepts shared predictor/retriever)
├── modules/
│   ├── initialization.py      # Dataset loader and seed chooser
│   ├── prediction.py          # XGBoost property prediction
│   ├── feasibility.py         # SAScore + valency checks
│   ├── scoring.py             # MAPE + feasibility combined score
│   ├── modification_tools.py  # Generic chemical transformations
│   ├── strategy_pool.py       # 81-tuple SMARTS strategies (validated)
│   ├── rag_retrieval.py       # RAG: name resolution, search, extraction
│   └── rag_cache.py           # SQLite cache for RAG results
├── evaluation/
│   └── rag_evaluation.py      # Precision@K / Recall@K for RAG
└── gui/                       # Flask + SSE web UI
```

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file (ignored by git) if you want LLM-based property extraction:

```
OPENAI_API_KEY=sk-...
```

LLM extraction is skipped gracefully when this key is absent.

## CLI usage

```bash
python main.py \
  --density 1.8 \
  --velocity 8500 \
  --pressure 35 \
  --hf -50 \
  --max-iter 10 \
  --output ./output/results.json
```

Add `--disable-rag` to skip literature retrieval (fast path, prediction-only).

## Python usage

```python
from data_structures import PropertyTarget
from config import Config
from designer import EnergeticDesigner

target = PropertyTarget(density=1.8, det_velocity=8500, det_pressure=35, hf_solid=-50)

config = Config()
config.rag.enable_rag = True
# Optional tuning:
# config.rag.cache_path = "./output/rag_cache.sqlite"
# config.rag.arxiv_max_results = 3

designer = EnergeticDesigner(target, config)
designer.initialize()
best = designer.run_design_loop()
print(best.smiles, best.score)
```

## GUI

```bash
python gui/app.py
# open http://localhost:5001
```

## Configuration highlights

- `beam_search.beam_width` / `top_k` / `max_iterations`
- `beam_search.convergence_threshold` — MAPE-% change below which the search is considered converged
- `scoring.mape_weight` / `sascore_weight` — combined-score weighting (default 0.7 / 0.3)
- `rag.enable_rag`, `rag.use_llm`, `rag.max_papers`, `rag.arxiv_max_results`, `rag.cache_path`, `rag.openai_api_key`

## Scoring

```
Total Score = mape_weight · MAPE(%) + sascore_weight · (1 − feasibility)
```

Lower is better.

## License

MIT
