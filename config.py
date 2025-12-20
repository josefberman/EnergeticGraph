"""
Configuration classes for the beam search molecular design system.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional

# Load environment variables from .env file
load_dotenv()


@dataclass
class BeamSearchConfig:
    """Configuration for beam search algorithm."""
    beam_width: int = 10  # Number of candidates to keep in beam
    top_k: int = 5  # Number of top candidates to select after each iteration
    max_iterations: int = 20  # Maximum number of search iterations
    convergence_threshold: float = 0.01  # Stop if improvement < threshold


@dataclass
class ScoringConfig:
    """Configuration for scoring function."""
    # Property weights for MAE calculation (should sum to 1.0)
    property_weights: dict = field(default_factory=lambda: {
        'Density': 0.25,
        'Det Velocity': 0.25,
        'Det Pressure': 0.25,
        'Hf solid': 0.25
    })
    
    # Multi-objective weights
    mae_weight: float = 0.7  # Weight for property accuracy
    feasibility_weight: float = 0.3  # Weight for synthetic feasibility
    
    # Property normalization ranges (for scaling)
    property_ranges: dict = field(default_factory=lambda: {
        'Density': (1.0, 2.5),  # g/cm³
        'Det Velocity': (6000.0, 10000.0),  # m/s
        'Det Pressure': (10.0, 50.0),  # GPa
        'Hf solid': (-500.0, 500.0)  # kJ/mol
    })


@dataclass
class RAGConfig:
    """Configuration for RAG-based modification strategy."""
    enable_rag: bool = True  # Enable/disable RAG
    arxiv_max_results: int = 5  # Max papers to retrieve per query
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))  # Auto-load from .env
    chroma_persist_directory: str = "./chroma_db"  # ChromaDB storage
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model
    llm_model: str = "gpt-4o-mini"  # ChatOpenAI model
    llm_temperature: float = 0.3  # LLM temperature
    max_modifications_per_call: int = 3  # Max SMILES to generate per RAG call


@dataclass
class SystemConfig:
    """System-level configuration."""
    models_directory: str = "./models"  # Path to XGBoost models
    dataset_path: str = "./sample_start_molecules.csv"  # Path to molecular dataset
    output_directory: str = "./output"  # Output results
    log_level: str = "INFO"  # Logging level
    random_seed: int = 42  # For reproducibility


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    beam_search: BeamSearchConfig = field(default_factory=BeamSearchConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
