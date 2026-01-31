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
    Adjusted for energetic materials which commonly have nitrogen-rich heterocycles.
    
    Returns score 1-10 (lower is better = easier to synthesize).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 10.0
        
        num_rings = Descriptors.RingCount(mol)
        num_rot_bonds = Descriptors.NumRotatableBonds(mol)
        num_atoms = mol.GetNumAtoms()
        num_heteroatoms = Descriptors.NumHeteroatoms(mol)
        
        # Count specific structural features
        num_nitrogens = smiles.count('N') + smiles.count('n')
        
        # Start with a base score (most molecules are synthesizable)
        score = 2.0
        
        # Penalize only very complex ring systems (>4 rings)
        if num_rings > 4:
            score += (num_rings - 4) * 0.4
        
        # Light penalty for many rotatable bonds (flexible = harder to control)
        score += min(num_rot_bonds * 0.15, 1.5)
        
        # Size penalty only for very large molecules (>30 atoms)
        if num_atoms > 30:
            score += min((num_atoms - 30) * 0.05, 1.5)
        
        # Heteroatom penalty is very low for energetic materials
        # (they commonly have many N, O atoms)
        if num_heteroatoms > 8:
            score += min((num_heteroatoms - 8) * 0.1, 1.0)
        
        # Check for problematic/unstable patterns
        unstable_patterns = [
            'O-O',  # Peroxides
            'N-N-N-N',  # Long nitrogen chains
            '[N-]=[N+]=N',  # Azides (slightly penalize)
        ]
        for pattern in unstable_patterns:
            if pattern in smiles:
                score += 0.3
        
        # Bonus for common energetic material scaffolds (well-studied synthesis)
        favorable_patterns = [
            'c1nnn',  # Tetrazole (well-known synthesis)
            'c1nn',   # Triazole
            'n1nnn',  # Tetrazole variant
            'c1ncn',  # Imidazole/pyrimidine
        ]
        for pattern in favorable_patterns:
            if pattern in smiles.lower():
                score -= 0.5
        
        return max(1.0, min(score, 10.0))
    
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
    Calculate normalized synthetic accessibility score.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Tuple of (normalized_sascore, is_feasible_bool)
        - normalized_sascore: 0-1 (0 = most feasible, 1 = least feasible)
          Used directly in combined score: 0.7*MAPE + 0.3*normalized_sascore
        - is_feasible_bool: True if passes basic checks (SAScore <= 7)
    """
    # Check valency first - invalid molecules get worst score
    if not check_valency(smiles):
        return 1.0, False
    
    # Calculate SAScore (1-10 scale, where 1 = easy, 10 = hard)
    sascore = calculate_sascore(smiles)
    
    # Normalize SAScore to 0-1 range (0 = most feasible, 1 = least feasible)
    # Linear normalization: (sascore - 1) / (10 - 1) = (sascore - 1) / 9
    normalized_sascore = (sascore - 1.0) / 9.0
    
    # Clamp to [0, 1] range
    normalized_sascore = max(0.0, min(1.0, normalized_sascore))
    
    # Consider feasible if SAScore <= 7 (moderately synthesizable)
    is_feasible = sascore <= 7.0
    
    return normalized_sascore, is_feasible
