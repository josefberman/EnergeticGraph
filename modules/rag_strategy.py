"""
Modification strategy module - now uses pre-built strategy pool.

This module provides molecular modification strategies for energetic materials.
The RAG-based approach has been replaced with a curated strategy pool.
"""

import logging
from typing import List, Dict

from .strategy_pool import StrategyPoolModifier, get_modification_strategies
from .modification_tools import generate_diverse_modifications

logger = logging.getLogger(__name__)


class ModificationStrategy:
    """
    Molecular modification strategy using pre-built strategy pool.
    
    Replaces the previous RAG-based approach (Arxiv + LLM) with
    instant, literature-backed modification patterns.
    """
    
    def __init__(self, config=None):
        """
        Initialize modification strategy.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.pool_modifier = StrategyPoolModifier(config)
        logger.info("Initialized ModificationStrategy with strategy pool")
    
    def get_modifications(self, smiles: str, property_gap: Dict[str, float], 
                         target_count: int = 10) -> List[str]:
        """
        Get molecular modifications based on property gaps.
        
        Args:
            smiles: Parent molecule SMILES
            property_gap: Dictionary of property gaps (target - current)
            target_count: Target number of modifications to generate
        
        Returns:
            List of modified SMILES strings
        """
        logger.info(f"Getting modifications for {smiles} with gaps: {property_gap}")
        
        # Use strategy pool
        modifications = self.pool_modifier.apply_strategies(
            smiles, property_gap, target_count
        )
        
        logger.info(f"Strategy pool returned {len(modifications)} modifications")
        return modifications


def default_modification_strategy(smiles: str, property_gap: Dict[str, float], 
                                  target_count: int = 20) -> List[str]:
    """
    Default modification strategy using strategy pool + diverse modifications.
    
    Args:
        smiles: Parent molecule SMILES
        property_gap: Property gaps (target - current)
        target_count: Target number of modifications (default 20)
    
    Returns:
        List of modified SMILES
    """
    logger.info(f"Using default modification strategy for {smiles} (target: {target_count})")
    
    # Primary: Use strategy pool
    modifications = get_modification_strategies(smiles, property_gap, target_count)
    
    # Supplement with diverse modifications if needed
    if len(modifications) < target_count:
        logger.info(f"Supplementing: have {len(modifications)}, need {target_count}")
        diverse_mods = generate_diverse_modifications(
            smiles, 
            target_count=(target_count - len(modifications)) * 2
        )
        
        # Add unique modifications
        existing = set(modifications)
        for mod in diverse_mods:
            if mod not in existing and mod != smiles and '.' not in mod:
                modifications.append(mod)
                existing.add(mod)
    
    # Remove duplicates and limit
    modifications = list(set(modifications))
    modifications = [m for m in modifications if m != smiles]
    
    logger.info(f"Default strategy generated {len(modifications)} candidates")
    return modifications[:target_count * 2]
