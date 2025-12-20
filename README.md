# Beam Search Molecular Design System

A modular system for designing energetic molecules using beam search optimization with RAG-enhanced modification strategies.

## Features

- **Beam Search Optimization**: Efficient exploration of chemical space
- **Property Prediction**: XGBoost models for Density, Detonation Velocity/Pressure, and Formation Enthalpy
- **Feasibility Checking**: SAScore-based synthetic accessibility
- **RAG-Powered Modifications**: LangChain + Arxiv integration for literature-informed molecular design
- **Modular Architecture**: Clean separation of concerns (prediction, modification, scoring, orchestration)

## Architecture

```
beam_search_system/
├── config.py                # Configuration classes
├── data_structures.py       # MoleculeState and PropertyTarget
├── descriptors.py           # Molecular descriptor generation
├── modules/
│   ├── initialization.py    # Dataset loader and seeder
│   ├── prediction.py        # XGBoost property prediction
│   ├── feasibility.py       # SAScore and valency checking
│   ├── scoring.py           # MAE + feasibility scoring
│   ├── modification_tools.py # Chemical modifications
│   └── rag_strategy.py      # RAG-based strategies
├── agents/
│   └── worker_agent.py      # ChemistAgent
├── orchestrator.py          # Beam Search Engine
├── designer.py              # Main EnergeticDesigner class
└── main.py                  # CLI entry point
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from beam_search_system.data_structures import PropertyTarget
from beam_search_system.config import Config
from beam_search_system.designer import EnergeticDesigner

# Define target properties
target = PropertyTarget(
    density=1.8,          # g/cm³
    det_velocity=8500.0,  # m/s
    det_pressure=35.0,    # GPa
    hf_solid=-50.0        # kJ/mol
)

# Create config
config = Config()
config.rag.enable_rag = True
config.rag.openai_api_key = "your-api-key"

# Initialize and run
designer = EnergeticDesigner(target, config)
designer.initialize()
best_molecule = designer.run_design_loop()

print(f"Best SMILES: {best_molecule.smiles}")
print(f"Score: {best_molecule.score:.4f}")
```

### Command Line

```bash
python beam_search_system/main.py \
  --density 1.8 \
  --velocity 8500 \
  --pressure 35 \
  --hf -50 \
  --enable-rag \
  --openai-key YOUR_API_KEY \
  --max-iter 10 \
  --output results.json
```

## Configuration

### Beam Search Parameters

- `beam_width`: Number of candidates in beam (default: 10)
- `top_k`: Top candidates to keep per iteration (default: 5)
- `max_iterations`: Maximum search iterations (default: 20)
- `convergence_threshold`: Stop if improvement < threshold (default: 0.01)

### RAG Parameters

- `enable_rag`: Enable/disable RAG (default: True)
- `arxiv_max_results`: Max papers per query (default: 5)
- `openai_api_key`: Your OpenAI API key
- `chroma_persist_directory`: ChromaDB storage path

## Scoring Function

```
Total Score = 0.7 × MAE + 0.3 × (1 - Feasibility)
```

Where:
- **MAE**: Normalized mean absolute error across properties
- **Feasibility**: SAScore-based synthetic accessibility (0-1)

Lower score is better.

## Modification Strategies

1. **RAG-Based (Primary)**: Queries Arxiv for literature-informed modifications
2. **Default Heuristics (Fallback)**: Rule-based chemical transformations
   - Addition: Add functional groups (-NO2, -NH2, -N3, -CN)
   - Subtraction: Remove terminal atoms/groups
   - Substitution: Replace atoms/groups
   - Ring Modifications: Add energetic rings (triazoles, tetrazoles)

## Requirements

- Python 3.8+
- RDKit
- XGBoost
- LangChain (OpenAI, Community, Experimental)
- ChromaDB
- Pandas, NumPy
- OpenAI API key (for RAG)

## License

MIT
