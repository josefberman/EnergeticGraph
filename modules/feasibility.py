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
    # More nuanced conversion:
    # SAScore 1-3: Easy synthesis (90-100% feasibility)
    # SAScore 3-5: Moderate (70-90% feasibility)  
    # SAScore 5-7: Challenging but doable (50-70% feasibility)
    # SAScore 7-10: Very difficult (0-50% feasibility)
    
    if sascore <= 3.0:
        feasibility_score = 0.90 + (3.0 - sascore) / 20.0  # 90-100%
    elif sascore <= 5.0:
        feasibility_score = 0.70 + (5.0 - sascore) * 0.10  # 70-90%
    elif sascore <= 7.0:
        feasibility_score = 0.50 + (7.0 - sascore) * 0.10  # 50-70%
    else:
        feasibility_score = max(0.0, 0.50 - (sascore - 7.0) * 0.167)  # 0-50%
    
    # Consider feasible if SAScore <= 7 (moderately synthesizable)
    is_feasible = sascore <= 7.0
    
    return feasibility_score, is_feasible
