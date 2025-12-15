"""Main molecular optimization agent - wrapper around working implementation.

This module provides the high-level API by wrapping the working beam search
implementation from molecular_optimizer_agent.py while keeping the modular
package structure.
"""
import sys
import os

# Add parent directory to path to import original working implementation
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the working implementation
from molecular_optimizer_agent import MolecularOptimizationAgent as _WorkingAgent

# Re-export with same interface
class MolecularOptimizationAgent(_WorkingAgent):
    """Enhanced molecular optimization agent using LangGraph.
    
    This wraps the working beam search implementation while providing
    a clean modular interface through the molecular_optimizer package.
    
    The modular components (feasibility, scoring, modifications, etc.) are
    available for standalone use, but the full optimization still uses the
    proven implementation.
    """
    pass


__all__ = ['MolecularOptimizationAgent']
