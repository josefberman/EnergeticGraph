"""State management for molecular optimization.

This module contains the core data structures used throughout
the molecular optimization process.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class OptimizationState:
    """State for the molecular optimization process."""
    current_molecule: str
    target_properties: Dict[str, float]
    weights: Dict[str, float]
    beam_candidates: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    beam_width: int
    best_score: float
    best_molecule: str
    best_gibbs: Optional[float]
    search_history: List[Dict[str, Any]]
    convergence_threshold: float
    verbose: bool


@dataclass
class FeasibilityReport:
    """Molecular feasibility assessment using ONLY SAScore.
    
    SAScore (Synthetic Accessibility Score):
    - Range: 1 (easy to synthesize) to 10 (very difficult)
    - Fast: ~1ms per molecule (1000x faster than xTB)
    - Based on real synthesis data and fragment analysis
    """
    # Synthetic accessibility (1=easy to synthesize, 10=very difficult)
    sa_score: float
    
    # Composite feasibility score (0-1, higher is better)
    # Directly derived from SAScore
    composite_score_0_1: float


