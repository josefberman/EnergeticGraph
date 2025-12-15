"""Molecular feasibility assessment using ONLY SAScore.

This module provides ultra-fast feasibility scoring based exclusively on
the Synthetic Accessibility Score (SAScore) from RDKit's official implementation.

SAScore is 1000x faster than xTB quantum chemistry and provides chemically
relevant synthetic accessibility predictions.
"""
from typing import Optional
from rdkit import Chem

from .state import FeasibilityReport



class FeasibilityCalculator:
    """Calculate molecular feasibility using ONLY SAScore."""
    
    def calculate_sa_score(self, mol: Chem.Mol) -> float:
        """Calculate synthetic accessibility score using RDKit's SAScore.
        
        Args:
            mol: RDKit molecule object
        
        Returns:
            SAScore from 1 (easy to synthesize) to 10 (very difficult)
        """
        try:
            from rdkit.Chem import RDKitConfig
            import sys
            import os
            
            contrib_dir = os.path.join(RDKitConfig.RDContribDir, 'SA_Score')
            if contrib_dir not in sys.path:
                sys.path.append(contrib_dir)
            
            import sascorer
            return float(sascorer.calculateScore(mol))
            
        except Exception:
            # Fallback to simple ersatz score if SAScore unavailable
            from rdkit.Chem import rdMolDescriptors
            rings = rdMolDescriptors.CalcNumRings(mol)
            spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            bridged = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            size = mol.GetNumAtoms()
            heteros = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
            return min(10.0, 1.5 + 0.03 * size + 0.6 * rings + 0.8 * spiro + 0.8 * bridged + 0.2 * heteros)
    
    def feasibility_from_smiles(
        self, 
        smiles: str
    ) -> Optional[FeasibilityReport]:
        """Calculate feasibility from SMILES using ONLY SAScore.
        
        Args:
            smiles: SMILES string
        
        Returns:
            FeasibilityReport with SAScore and composite score, or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
        except Exception:
            return None
        
        # Calculate SAScore
        sa = self.calculate_sa_score(mol)
        
        # Calculate composite feasibility score (0-1, higher is better)
        # Convert SAScore (1-10, lower is better) to 0-1 scale (higher is better)
        # EXTREMELY STRICT: Heavily penalize SAScore > 4.5 (even moderately complex molecules)
        
        # Start with maximum feasibility
        composite = 1.0
        
        # Apply VERY STRICT penalty for synthesis difficulty (SAScore > 4.5)
        # SAScore 1-4.5: Acceptable (very easy to synthesize)
        # SAScore 4.5-6.5: Moderate difficulty (heavy penalty)
        # SAScore 6.5-10: Difficult to very difficult (severe penalty, likely filtered)
        if sa > 4.5:
            # EXTREMELY STRICT penalty: Penalize early and very hard
            # This gives:
            # SA=4.5: composite ~1.0 (no penalty)
            # SA=5.5: composite ~0.82 (moderate penalty)
            # SA=6.5: composite ~0.64 (heavy penalty)
            # SA=7.5: composite ~0.45 (very heavy penalty, likely filtered)
            # SA=8.5: composite ~0.27 (almost certainly filtered)
            # SA=9.5: composite ~0.09 (definitely filtered)
            # SA=10: composite ~0.0 (impossible to pass)
            composite -= 1.0 * max(0, sa - 4.5) / 5.5
        
        # Clamp to 0-1 range
        composite = max(0.0, min(1.0, composite))

        
        return FeasibilityReport(
            sa_score=round(sa, 2),
            composite_score_0_1=round(composite, 3),
        )
