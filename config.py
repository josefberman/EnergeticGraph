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
    convergence_threshold: float = 0.001  # Stop if improvement < threshold


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
    
    # Maximum normalized SAScore threshold for candidates (0.67 ≈ SAScore 7)
    max_sascore: float = 0.67


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
    strategy_pool: StrategyPoolConfig = field(default_factory=StrategyPoolConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Legacy compatibility - kept for any code that references config.rag
    @property
    def rag(self):
        """Legacy property for backward compatibility."""
        return self.strategy_pool
