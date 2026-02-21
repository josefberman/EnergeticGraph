"""
Evaluation module for the Energetic Molecular Design System.

Contains evaluation scripts for:
- RAG module performance (Precision@K, Recall@K)
"""

from .rag_evaluation import run_evaluation, GROUND_TRUTH_DATA

__all__ = ['run_evaluation', 'GROUND_TRUTH_DATA']
