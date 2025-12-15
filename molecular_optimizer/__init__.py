"""Molecular Optimizer Package.

A modular architecture for molecular optimization using beam search,
RAG integration, and feasibility scoring.
"""
from .agent import MolecularOptimizationAgent
from .state import OptimizationState, FeasibilityReport
from .feasibility import FeasibilityCalculator
from .scoring import ScoringCalculator
from .beam_search import BeamSearchOptimizer
from .modifications import MolecularModifier
from .rag_integration import RAGIntegration

__version__ = '2.0.0'

__all__ = [
    'MolecularOptimizationAgent',
    'OptimizationState',
    'FeasibilityReport',
    'FeasibilityCalculator',
    'ScoringCalculator',
    'BeamSearchOptimizer',
    'MolecularModifier',
    'RAGIntegration',
]
