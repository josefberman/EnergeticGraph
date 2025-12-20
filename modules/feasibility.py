"""
Feasibility checking module using SAScore and RDKit validation.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def calculate_sascore(smiles: str) -> float:
    """
    Calculate Synthetic Accessibility Score (SAScore).
    
    Args:
        smiles: SMILES string
        
    Returns:
        SAScore (1-10, where 1 is easy to synthesize, 10 is very difficult)
        Returns 10.0 (worst) if calculation fails
    """
    try:
        from rdkit.Chem import RDConfig
        import sys
        import os
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        
        score = sascorer.calculateScore(mol)
        return float(score)
    
    except ImportError:
        # If SA_Score not available, use a simpler heuristic
        logger.warning("sascorer not available, using simple heuristic")
        return _simple_sascore_estimate(smiles)
    except Exception as e:
        logger.error(f"Error calculating SAScore for {smiles}: {e}")
        return 10.0


def _simple_sascore_estimate(smiles: str) -> float:
    """
    Simple SAScore estimate based on molecular complexity.
    
    Returns score 1-10 (lower is better).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        
        # Simple heuristic based on rings, rotatable bonds, and complexity
        num_rings = Descriptors.RingCount(mol)
        num_rot_bonds = Descriptors.NumRotatableBonds(mol)
        num_atoms = mol.GetNumAtoms()
        num_heteroatoms = Descriptors.NumHeteroatoms(mol)
        
        # Penalize: many rings, many rotatable bonds, complex structures
        score = 1.0
        score += min(num_rings * 0.5, 3.0)
        score += min(num_rot_bonds * 0.3, 2.0)
        score += min((num_atoms - 10) * 0.1, 2.0) if num_atoms > 10 else 0
        score += min(num_heteroatoms * 0.2, 2.0)
        
        return min(score, 10.0)
    
    except:
        return 10.0


def check_valency(smiles: str) -> bool:
    """
    Check if molecule has valid valency using RDKit sanitization.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Try to sanitize - will fail if valency is wrong
        Chem.SanitizeMol(mol)
        return True
    
    except:
        return False


def calculate_feasibility(smiles: str) -> Tuple[float, bool]:
    """
    Calculate overall feasibility score.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Tuple of (feasibility_score, is_feasible_bool)
        - feasibility_score: 0-1 (higher is better)
        - is_feasible_bool: True if passes basic checks
    """
    # Check valency first
    if not check_valency(smiles):
        return 0.0, False
    
    # Calculate SAScore
    sascore = calculate_sascore(smiles)
    
    # Convert SAScore (1-10, lower is better) to feasibility (0-1, higher is better)
    # Use a cutoff: SAScore > 6 is considered difficult
    feasibility_score = max(0.0, (10.0 - sascore) / 10.0)
    
    # Consider feasible if SAScore <= 7 (moderately easy to synthesize)
    is_feasible = sascore <= 7.0
    
    return feasibility_score, is_feasible
