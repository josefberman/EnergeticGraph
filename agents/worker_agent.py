"""
Worker Agent (ChemistAgent) - Generates molecular variations.
"""

import logging
from typing import List, Dict
from rdkit import Chem
from data_structures import MoleculeState, PropertyTarget
from modules.prediction import PropertyPredictor
from modules.feasibility import calculate_feasibility
from modules.scoring import calculate_total_score
from modules.strategy_pool import StrategyPoolModifier, default_modification_strategy
from config import Config

logger = logging.getLogger(__name__)


class ChemistAgent:
    """
    Worker agent that generates molecular variations using strategy pool.
    """
    
    def __init__(self, parent_molecule: MoleculeState, 
                 target_properties: PropertyTarget,
                 config: Config):
        """
        Initialize chemist agent.
        
        Args:
            parent_molecule: Parent MoleculeState
            target_properties: Target properties
            config: System configuration
        """
        self.parent = parent_molecule
        self.target = target_properties
        self.config = config
        
        # Initialize components
        self.predictor = PropertyPredictor(config.system.models_directory)
        
        # Initialize strategy pool modifier
        self.strategy_modifier = StrategyPoolModifier(config)
        logger.info("Initialized ChemistAgent with strategy pool")
    
    def analyze_property_gap(self) -> Dict[str, float]:
        """
        Calculate property gaps (target - current).
        
        Returns:
            Dictionary of property gaps
        """
        target_dict = self.target.to_dict()
        parent_props = self.parent.properties
        
        gaps = {}
        for prop_name, target_val in target_dict.items():
            if prop_name in parent_props and parent_props[prop_name] is not None:
                current_val = parent_props[prop_name]
                gaps[prop_name] = target_val - current_val
            else:
                gaps[prop_name] = 0.0
        
        logger.debug(f"Property gaps: {gaps}")
        return gaps
    
    def generate_variations(self) -> List[MoleculeState]:
        """
        Generate molecular variations using strategy pool.
        
        Workflow:
        1. Analyze property gap
        2. Use strategy pool to get targeted modifications
        3. Supplement with default strategy if needed
        4. Evaluate all candidates
        5. Return up to beam_width MoleculeState objects
        
        Returns:
            List of candidate MoleculeState objects
        """
        target_count = self.config.beam_search.beam_width
        
        # 1. Analyze property gap
        property_gap = self.analyze_property_gap()
        
        # 2. Collect candidate SMILES
        candidates = []
        seen = set()
        max_attempts = 5
        attempt = 0
        
        logger.info(f"Generating candidates for beam_width={target_count}")
        
        while len(candidates) < target_count and attempt < max_attempts:
            attempt += 1
            needed = target_count - len(candidates)
            logger.info(f"Attempt {attempt}/{max_attempts}: Need {needed} more (have {len(candidates)})")
            
            # Request more than we need to account for filtering
            request_count = max(needed * 3, target_count * 2)
            batch_smiles = []
            
            # PRIMARY: Strategy pool modifications
            try:
                pool_smiles = self.strategy_modifier.apply_strategies(
                    self.parent.smiles,
                    property_gap,
                    target_count=request_count
                )
                batch_smiles.extend(pool_smiles)
                logger.info(f"Strategy pool returned {len(pool_smiles)} modifications")
            except Exception as e:
                logger.warning(f"Strategy pool failed: {e}")
            
            # FALLBACK: Default strategy
            if len(batch_smiles) < needed:
                try:
                    default_smiles = default_modification_strategy(
                        self.parent.smiles,
                        property_gap,
                        target_count=request_count
                    )
                    batch_smiles.extend(default_smiles)
                    logger.info(f"Default strategy added {len(default_smiles)} modifications")
                except Exception as e:
                    logger.warning(f"Default strategy failed: {e}")
            
            # ADDITIONAL: Direct diverse modifications
            if len(batch_smiles) < needed:
                from modules.modification_tools import generate_diverse_modifications
                try:
                    diverse_smiles = generate_diverse_modifications(
                        self.parent.smiles, 
                        target_count=request_count
                    )
                    batch_smiles.extend(diverse_smiles)
                    logger.info(f"Diverse generation added {len(diverse_smiles)} modifications")
                except Exception as e:
                    logger.warning(f"Diverse generation failed: {e}")
            
            # Filter to unique unseen SMILES
            unique_batch = []
            for s in batch_smiles:
                if s not in seen and s != self.parent.smiles and '.' not in s:
                    seen.add(s)
                    unique_batch.append(s)
            
            logger.info(f"Unique new candidates to evaluate: {len(unique_batch)}")
            
            # Evaluate candidates
            for smiles in unique_batch:
                if len(candidates) >= target_count:
                    break
                candidate = self.evaluate_candidate(smiles)
                if candidate is not None:
                    candidates.append(candidate)
            
            # Recursive expansion from existing candidates if needed
            if len(candidates) < target_count and len(candidates) > 0:
                logger.info(f"Expanding from {len(candidates)} existing candidates")
                from modules.modification_tools import apply_all_modifications
                
                for existing in candidates[:5]:  # Limit expansion base
                    if len(candidates) >= target_count:
                        break
                    try:
                        expanded = apply_all_modifications(existing.smiles)
                        for exp_smiles in expanded:
                            if exp_smiles not in seen and '.' not in exp_smiles:
                                seen.add(exp_smiles)
                                exp_candidate = self.evaluate_candidate(exp_smiles)
                                if exp_candidate is not None:
                                    candidates.append(exp_candidate)
                                    if len(candidates) >= target_count:
                                        break
                    except Exception as e:
                        logger.debug(f"Expansion failed: {e}")
        
        if not candidates:
            logger.warning(f"No valid candidates generated for {self.parent.smiles}")
            return []
        
        final_candidates = candidates[:target_count]
        logger.info(f"Returning {len(final_candidates)} candidates (target: {target_count})")
        return final_candidates
    
    def evaluate_candidate(self, smiles: str) -> MoleculeState:
        """
        Evaluate a candidate molecule.
        
        Args:
            smiles: Candidate SMILES string
        
        Returns:
            MoleculeState object or None if invalid
        """
        try:
            # Skip multi-molecule SMILES
            if '.' in smiles:
                return None
            
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.debug(f"Invalid SMILES: {smiles}")
                return None
            
            # Canonicalize
            smiles = Chem.MolToSmiles(mol)
            
            # Predict properties
            predicted_props = self.predictor.predict_properties(smiles)
            if predicted_props is None:
                logger.debug(f"Failed to predict properties for {smiles}")
                return None
            
            # Calculate feasibility
            feasibility_score, is_feasible = calculate_feasibility(smiles)
            
            # Calculate total score
            target_dict = self.target.to_dict()
            score = calculate_total_score(
                predicted_props,
                target_dict,
                feasibility_score,
                mape_weight=self.config.scoring.mape_weight,
                feasibility_weight=self.config.scoring.feasibility_weight,
                property_weights=self.config.scoring.property_weights
            )
            
            # Create MoleculeState
            candidate = MoleculeState(
                smiles=smiles,
                properties=predicted_props,
                score=score,
                feasibility=feasibility_score,
                is_feasible=is_feasible,
                provenance=f"{self.parent.provenance} -> modification",
                generation=self.parent.generation + 1,
                parent_smiles=self.parent.smiles
            )
            
            logger.debug(f"Evaluated {smiles}: score={score:.4f}, feasible={is_feasible}")
            return candidate
        
        except Exception as e:
            logger.error(f"Error evaluating candidate {smiles}: {e}")
            return None
