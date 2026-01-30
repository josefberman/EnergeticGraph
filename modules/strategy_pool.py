"""
Strategy Pool - Pre-built molecular modification strategies based on literature.

Contains SMARTS reaction patterns for modifying energetic materials
to achieve specific property changes (increase/decrease density,
detonation velocity, detonation pressure, heat of formation).
"""

import logging
from typing import List, Dict, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

from .modification_tools import generate_diverse_modifications

logger = logging.getLogger(__name__)


# =============================================================================
# STRATEGY POOL - Literature-based modification strategies
# =============================================================================
# Each strategy contains:
#   - SMARTS reaction pattern
#   - Description
#   - Property effects: (density, det_velocity, det_pressure, hf_solid)
#     where effect is: +1 (increase), -1 (decrease), 0 (neutral/minor)
# =============================================================================

MODIFICATION_STRATEGIES = {
    # =========================================================================
    # NITRO GROUP ADDITIONS - Generally increase all energetic properties
    # =========================================================================
    'add_nitro_aromatic': {
        'smarts': '[c:1][H]>>[c:1][N+](=O)[O-]',
        'description': 'Add nitro group to aromatic carbon',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +1},
        'references': ['Klapötke 2017', 'Politzer 2004'],
    },
    'add_nitro_aliphatic': {
        'smarts': '[C:1]([H])([H])[H]>>[C:1]([H])([H])[N+](=O)[O-]',
        'description': 'Add nitro group to methyl carbon',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +1},
        'references': ['Politzer 2004'],
    },
    
    # =========================================================================
    # AZIDO GROUP ADDITIONS - High nitrogen content, increases Hf significantly
    # =========================================================================
    'add_azido_aromatic': {
        'smarts': '[c:1][H]>>[c:1]N=[N+]=[N-]',
        'description': 'Add azido group to aromatic carbon',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +2},
        'references': ['Klapötke 2012'],
    },
    'add_azido_aliphatic': {
        'smarts': '[C:1]([H])([H])[H]>>[C:1]([H])([H])N=[N+]=[N-]',
        'description': 'Add azido group to methyl carbon',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +2},
        'references': ['Klapötke 2012'],
    },
    
    # =========================================================================
    # NITRAMINE FORMATION - Convert amines to nitramines
    # =========================================================================
    'nitramine_formation': {
        'smarts': '[N:1]([H])[H]>>[N:1]([H])[N+](=O)[O-]',
        'description': 'Convert primary amine to nitramine (N-NO2)',
        'effects': {'Density': +1, 'Det Velocity': +2, 'Det Pressure': +2, 'Hf solid': +1},
        'references': ['RDX/HMX chemistry', 'Klapötke 2017'],
    },
    'secondary_nitramine': {
        'smarts': '[N:1]([H])([C:2])[C:3]>>[N:1]([N+](=O)[O-])([C:2])[C:3]',
        'description': 'Convert secondary amine to nitramine',
        'effects': {'Density': +1, 'Det Velocity': +2, 'Det Pressure': +2, 'Hf solid': +1},
        'references': ['RDX/HMX chemistry'],
    },
    
    # =========================================================================
    # NITROGEN-RICH HETEROCYCLES - High Hf, good density
    # =========================================================================
    'add_tetrazole': {
        'smarts': '[C:1]#N>>[C:1]c1nnn[nH]1',
        'description': 'Convert cyano to tetrazole ring',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +2},
        'references': ['Tetrazole energetics literature'],
    },
    'add_triazole_aromatic': {
        'smarts': '[c:1][H]>>[c:1]c1cn[nH]n1',
        'description': 'Add 1,2,3-triazole to aromatic carbon',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +2},
        'references': ['Click chemistry energetics'],
    },
    'add_furazan': {
        'smarts': '[c:1][H]>>[c:1]c1nonc1',
        'description': 'Add furazan (1,2,5-oxadiazole) ring',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +1},
        'references': ['Furazan-based explosives'],
    },
    
    # =========================================================================
    # N-OXIDE FORMATION - Increases oxygen balance and density
    # =========================================================================
    'n_oxide_pyridine': {
        'smarts': '[n:1]>>[n+:1][O-]',
        'description': 'N-oxide formation on pyridine-type nitrogen',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': 0},
        'references': ['N-oxide energetics'],
    },
    
    # =========================================================================
    # HALOGENATION - Increases density, varies other effects
    # =========================================================================
    'add_fluoro_aromatic': {
        'smarts': '[c:1][H]>>[c:1]F',
        'description': 'Add fluorine to aromatic carbon',
        'effects': {'Density': +1, 'Det Velocity': 0, 'Det Pressure': 0, 'Hf solid': -1},
        'references': ['Fluorinated energetics'],
    },
    'add_chloro_aromatic': {
        'smarts': '[c:1][H]>>[c:1]Cl',
        'description': 'Add chlorine to aromatic carbon',
        'effects': {'Density': +1, 'Det Velocity': -1, 'Det Pressure': 0, 'Hf solid': -1},
        'references': ['Halogenated compounds'],
    },
    
    # =========================================================================
    # AMINO GROUPS - Can be further nitrated, good precursors
    # =========================================================================
    'add_amino_aromatic': {
        'smarts': '[c:1][H]>>[c:1]N',
        'description': 'Add amino group to aromatic carbon',
        'effects': {'Density': 0, 'Det Velocity': 0, 'Det Pressure': 0, 'Hf solid': +1},
        'references': ['Amino energetics'],
    },
    
    # =========================================================================
    # HYDROXYL GROUPS - Hydrogen bonding, crystal density
    # =========================================================================
    'add_hydroxyl_aromatic': {
        'smarts': '[c:1][H]>>[c:1]O',
        'description': 'Add hydroxyl group to aromatic carbon',
        'effects': {'Density': +1, 'Det Velocity': 0, 'Det Pressure': 0, 'Hf solid': -1},
        'references': ['Phenolic energetics'],
    },
    
    # =========================================================================
    # CYANO GROUPS - Moderate energy, good for further transformations
    # =========================================================================
    'add_cyano_aromatic': {
        'smarts': '[c:1][H]>>[c:1]C#N',
        'description': 'Add cyano group to aromatic carbon',
        'effects': {'Density': +1, 'Det Velocity': 0, 'Det Pressure': 0, 'Hf solid': +1},
        'references': ['Nitrile energetics'],
    },
    
    # =========================================================================
    # METHYLATION - Generally decreases density (adds CH3)
    # =========================================================================
    'add_methyl_aromatic': {
        'smarts': '[c:1][H]>>[c:1]C',
        'description': 'Add methyl group to aromatic carbon',
        'effects': {'Density': -1, 'Det Velocity': -1, 'Det Pressure': -1, 'Hf solid': 0},
        'references': ['General organic chemistry'],
    },
    
    # =========================================================================
    # RING FUSION - Increases density and rigidity
    # =========================================================================
    'benzene_fusion': {
        'smarts': '[c:1]1[c:2][c:3][c:4][c:5][c:6]1>>[c:1]1[c:2][c:3][c:4]2[c:5][c:6]1cccc2',
        'description': 'Fuse benzene ring to form naphthalene',
        'effects': {'Density': +1, 'Det Velocity': 0, 'Det Pressure': +1, 'Hf solid': +1},
        'references': ['PAH energetics'],
    },
    
    # =========================================================================
    # SUBSTITUTIONS - Replace atoms
    # =========================================================================
    'replace_ch_with_n': {
        'smarts': '[c:1][H]>>[n:1]',
        'description': 'Replace aromatic CH with N (pyridine formation)',
        'effects': {'Density': +1, 'Det Velocity': +1, 'Det Pressure': +1, 'Hf solid': +1},
        'references': ['Nitrogen heterocycles'],
    },
    'replace_ch3_with_no2': {
        'smarts': '[C:1]([c:2])([H])([H])[H]>>[c:2][N+](=O)[O-]',
        'description': 'Replace methyl with nitro group',
        'effects': {'Density': +2, 'Det Velocity': +2, 'Det Pressure': +2, 'Hf solid': +1},
        'references': ['Nitration chemistry'],
    },
    
    # =========================================================================
    # REMOVAL STRATEGIES - For decreasing properties
    # =========================================================================
    'remove_nitro': {
        'smarts': '[c:1][N+](=O)[O-]>>[c:1][H]',
        'description': 'Remove nitro group from aromatic ring',
        'effects': {'Density': -1, 'Det Velocity': -1, 'Det Pressure': -1, 'Hf solid': -1},
        'references': ['Denitration'],
    },
    'remove_halogen_add_h': {
        'smarts': '[c:1][F,Cl,Br,I]>>[c:1][H]',
        'description': 'Remove halogen, add hydrogen',
        'effects': {'Density': -1, 'Det Velocity': 0, 'Det Pressure': 0, 'Hf solid': 0},
        'references': ['Dehalogenation'],
    },
}


# =============================================================================
# PROPERTY-SPECIFIC STRATEGY GROUPS
# =============================================================================

STRATEGIES_BY_GOAL = {
    # Strategies for INCREASING properties
    'increase_Density': [
        'add_nitro_aromatic', 'add_nitro_aliphatic', 'add_azido_aromatic',
        'add_fluoro_aromatic', 'add_chloro_aromatic', 'nitramine_formation',
        'add_tetrazole', 'add_furazan', 'n_oxide_pyridine', 'replace_ch_with_n',
        'replace_ch3_with_no2', 'add_hydroxyl_aromatic', 'add_cyano_aromatic',
    ],
    'increase_Det Velocity': [
        'add_nitro_aromatic', 'add_nitro_aliphatic', 'nitramine_formation',
        'secondary_nitramine', 'add_azido_aromatic', 'add_azido_aliphatic',
        'add_tetrazole', 'add_triazole_aromatic', 'add_furazan', 'n_oxide_pyridine',
        'replace_ch_with_n', 'replace_ch3_with_no2',
    ],
    'increase_Det Pressure': [
        'add_nitro_aromatic', 'add_nitro_aliphatic', 'nitramine_formation',
        'secondary_nitramine', 'add_azido_aromatic', 'add_tetrazole',
        'benzene_fusion', 'replace_ch_with_n', 'replace_ch3_with_no2',
    ],
    'increase_Hf solid': [
        'add_azido_aromatic', 'add_azido_aliphatic', 'add_tetrazole',
        'add_triazole_aromatic', 'nitramine_formation', 'add_nitro_aromatic',
        'add_amino_aromatic', 'add_cyano_aromatic', 'replace_ch_with_n',
        'benzene_fusion',
    ],
    
    # Strategies for DECREASING properties
    'decrease_Density': [
        'add_methyl_aromatic', 'remove_halogen_add_h', 'remove_nitro',
    ],
    'decrease_Det Velocity': [
        'remove_nitro', 'add_methyl_aromatic', 'add_chloro_aromatic',
    ],
    'decrease_Det Pressure': [
        'remove_nitro', 'add_methyl_aromatic',
    ],
    'decrease_Hf solid': [
        'add_fluoro_aromatic', 'add_chloro_aromatic', 'add_hydroxyl_aromatic',
        'remove_nitro',
    ],
}


class StrategyPoolModifier:
    """
    Applies pre-built molecular modification strategies based on property gaps.
    
    Replaces RAG-based dynamic strategy lookup with instant, literature-backed
    modification patterns.
    """
    
    def __init__(self, config=None):
        """
        Initialize strategy pool modifier.
        
        Args:
            config: Optional configuration object (for compatibility)
        """
        self.config = config
        self.strategies = MODIFICATION_STRATEGIES
        self.strategies_by_goal = STRATEGIES_BY_GOAL
        logger.info(f"Initialized StrategyPoolModifier with {len(self.strategies)} strategies")
    
    def get_strategies_for_gap(self, property_gap: Dict[str, float]) -> List[str]:
        """
        Select appropriate strategies based on property gaps.
        
        Args:
            property_gap: Dictionary of {property_name: gap_value}
                         Positive gap = need to increase, negative = decrease
        
        Returns:
            List of strategy names to apply
        """
        selected = set()
        
        for prop_name, gap in property_gap.items():
            if abs(gap) < 0.01:
                # Gap is negligible, skip
                continue
            
            if gap > 0:
                # Need to INCREASE this property
                key = f'increase_{prop_name}'
            else:
                # Need to DECREASE this property
                key = f'decrease_{prop_name}'
            
            if key in self.strategies_by_goal:
                # Add all strategies for this goal
                selected.update(self.strategies_by_goal[key])
        
        # If no specific strategies selected, use general enhancement
        if not selected:
            selected.update([
                'add_nitro_aromatic', 'add_azido_aromatic', 
                'add_amino_aromatic', 'add_tetrazole'
            ])
        
        logger.info(f"Selected {len(selected)} strategies for property gaps: {property_gap}")
        return list(selected)
    
    def apply_strategy(self, smiles: str, strategy_name: str) -> List[str]:
        """
        Apply a single strategy to a molecule.
        
        Args:
            smiles: Input SMILES string
            strategy_name: Name of strategy to apply
        
        Returns:
            List of modified SMILES strings
        """
        if strategy_name not in self.strategies:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return []
        
        strategy = self.strategies[strategy_name]
        smarts = strategy['smarts']
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        results = []
        
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                logger.debug(f"Failed to parse SMARTS for {strategy_name}")
                return []
            
            products = rxn.RunReactants((mol,))
            
            for product_tuple in products:
                for product in product_tuple:
                    try:
                        Chem.SanitizeMol(product)
                        new_smiles = Chem.MolToSmiles(product)
                        
                        # Skip fragmented molecules
                        if '.' in new_smiles:
                            continue
                        
                        # Skip if same as input
                        if new_smiles == smiles:
                            continue
                        
                        # Validate
                        if Chem.MolFromSmiles(new_smiles) is not None:
                            results.append(new_smiles)
                            
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"Strategy {strategy_name} failed: {e}")
        
        return results
    
    def apply_strategies(self, smiles: str, property_gap: Dict[str, float], 
                        target_count: int = 10) -> List[str]:
        """
        Apply all relevant strategies to generate modified molecules.
        
        Args:
            smiles: Parent molecule SMILES
            property_gap: Dictionary of property gaps
            target_count: Target number of modifications to generate
        
        Returns:
            List of unique modified SMILES strings
        """
        logger.info(f"Applying strategy pool to {smiles} (target: {target_count})")
        
        # Get relevant strategies
        strategy_names = self.get_strategies_for_gap(property_gap)
        
        all_modifications = set()
        
        # Apply each strategy
        for strategy_name in strategy_names:
            if len(all_modifications) >= target_count * 2:
                break
                
            new_mods = self.apply_strategy(smiles, strategy_name)
            all_modifications.update(new_mods)
            
            if new_mods:
                logger.debug(f"Strategy {strategy_name}: +{len(new_mods)} modifications")
        
        logger.info(f"Strategy pool generated {len(all_modifications)} modifications")
        
        # If not enough, supplement with diverse modifications
        if len(all_modifications) < target_count:
            logger.info(f"Supplementing with diverse modifications")
            diverse_mods = generate_diverse_modifications(
                smiles, 
                target_count=target_count - len(all_modifications)
            )
            # Filter out fragmented molecules
            for mod in diverse_mods:
                if '.' not in mod and mod != smiles:
                    all_modifications.add(mod)
        
        result = list(all_modifications)[:target_count]
        logger.info(f"Returning {len(result)} total modifications")
        return result


def get_modification_strategies(smiles: str, property_gap: Dict[str, float], 
                                target_count: int = 20) -> List[str]:
    """
    Convenience function to get modifications using strategy pool.
    
    Args:
        smiles: Parent molecule SMILES
        property_gap: Property gaps (target - current)
        target_count: Target number of modifications
    
    Returns:
        List of modified SMILES
    """
    modifier = StrategyPoolModifier()
    return modifier.apply_strategies(smiles, property_gap, target_count)
