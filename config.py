"""
Configuration classes for the beam search molecular design system.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Chat model when backend is ``ollama`` (`OLLAMA_MODEL` overrides); GUI can change each run.
# Library default tag is often ``qwen3.5:latest``; the OpenAI-compat ``/v1`` calls work with ``qwen3.5``.
DEFAULT_OLLAMA_MODEL = "qwen3.5"

# Default Ollama host when ``OLLAMA_BASE_URL`` is unset or empty.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Fixed analogue-ranking embeddings (ChemBERT-style classifier backbone; mean pool + L2 norm).
LITERATURE_EMBEDDING_MODEL = "xluobd/chemberta-iupac-classifier"


def _default_literature_llm_backend() -> str:
    """Chat backend for extraction: ``openai`` or ``ollama`` (legacy ``huggingface`` → ``openai``)."""
    raw = (os.getenv('LITERATURE_LLM_BACKEND', 'openai') or 'openai').strip().lower()
    if raw == 'huggingface':
        return 'openai'
    if raw in ('openai', 'ollama'):
        return raw
    return 'openai'


def _default_ollama_base_url() -> Optional[str]:
    raw = os.getenv('OLLAMA_BASE_URL')
    if raw is None:
        return DEFAULT_OLLAMA_BASE_URL
    stripped = raw.strip()
    return stripped if stripped else DEFAULT_OLLAMA_BASE_URL


@dataclass
class BeamSearchConfig:
    """Configuration for beam search algorithm."""
    beam_width: int = 10
    top_k: int = 5
    max_iterations: int = 20

    # Absolute MAPE target (%). Search stops when best MAPE ≤ this value.
    # 0 disables the check (run until max_iterations or no-improvement).
    mape_target: float = 0.0

    # Minimum MAPE-improvement (percentage points) between iterations.
    # If improvement < this for `patience` consecutive iterations, stop.
    convergence_threshold: float = 0.05
    patience: int = 2


@dataclass
class ScoringConfig:
    """Configuration for scoring function."""
    # Property weights for MAPE calculation (should sum to 1.0)
    property_weights: dict = field(default_factory=lambda: {
        'Density': 0.25,
        'Det Velocity': 0.25,
        'Det Pressure': 0.25,
        'Hf solid': 0.25
    })
    
    # Multi-objective weights for combined score: mape_weight * MAPE + sascore_weight * SAScore
    mape_weight: float = 0.7  # Weight for property accuracy (MAPE)
    sascore_weight: float = 0.3  # Weight for normalized SAScore (0=feasible, 1=infeasible)


@dataclass
class StrategyPoolConfig:
    """Configuration for strategy pool-based modifications."""
    # Maximum modifications per strategy application
    max_modifications_per_strategy: int = 10
    
    # Enable supplementary diverse modifications
    enable_diverse_supplement: bool = True
    
    # Maximum normalized SAScore threshold for candidates (0.7 ≈ SAScore 7)
    max_sascore: float = 0.7


@dataclass
class LiteratureSearchConfig:
    """Configuration for literature-based property lookup."""
    enable_literature_search: bool = True
    use_llm: bool = False
    max_papers: int = 3
    timeout: int = 15

    #: Chat backend for extraction: ``openai`` or ``ollama``.
    llm_backend: str = field(default_factory=_default_literature_llm_backend)

    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv('OPENAI_API_KEY')
    )

    cache_path: Optional[str] = "./output/literature_cache.sqlite"

    ollama_base_url: Optional[str] = field(default_factory=_default_ollama_base_url)
    #: Ollama chat tag (property extraction / LLM analogues); see ``LITERATURE_EMBEDDING_MODEL`` for embeddings.
    ollama_model: str = field(
        default_factory=lambda: os.getenv('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL)
    )


@dataclass
class SystemConfig:
    """System-level configuration."""
    models_directory: str = "./models"  # Path to XGBoost models
    dataset_path: str = "./sample_start_molecules.csv"  # Path to molecular dataset
    output_directory: str = "./output"  # Output results
    log_level: str = "WARNING"  # Logging level (WARNING = clean output, INFO/DEBUG = verbose)
    random_seed: int = 42  # For reproducibility


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    beam_search: BeamSearchConfig = field(default_factory=BeamSearchConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    strategy_pool: StrategyPoolConfig = field(default_factory=StrategyPoolConfig)
    literature: LiteratureSearchConfig = field(default_factory=LiteratureSearchConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
