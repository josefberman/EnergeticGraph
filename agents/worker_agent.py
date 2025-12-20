"""
Worker Agent (ChemistAgent) - Generates molecular variations.
"""

import logging
from typing import List, Dict
from data_structures import MoleculeState, PropertyTarget
from modules.prediction import PropertyPredictor
from modules.feasibility import calculate_feasibility
from modules.scoring import calculate_total_score
from modules.rag_strategy import RAGModificationStrategy, default_modification_strategy
from config import Config

logger = logging.getLogger(__name__)


class ChemistAgent:
    """
    Worker agent that generates molecular variations using RAG or default strategies.
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
        
        # Initialize RAG strategy if enabled
        self.rag_strategy = None
        if config.rag.enable_rag:
            try:
                self.rag_strategy = RAGModificationStrategy(config.rag)
                logger.info("RAG strategy initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG: {e}. Will use default strategy.")
    
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
        Generate molecular variations using RAG (if enabled) or default strategy.
        
        Workflow:
        1. Analyze property gap
        2. Try RAG strategy if enabled to get up to beam_width modifications
        3. Fill remaining slots with default strategy if needed
        4. Evaluate all candidates
        5. Return exactly beam_width MoleculeState objects (or fewer if evaluation fails)
        
        Returns:
            List of candidate MoleculeState objects (target: beam_width candidates)
        """
        target_count = self.config.beam_search.beam_width
        
        # 1. Analyze property gap
        property_gap = self.analyze_property_gap()
        
        # 2. Generate modifications using RAG first
        modified_smiles = []
        
        if self.config.rag.enable_rag and self.rag_strategy is not None:
            logger.info(f"Attempting RAG-based modification strategy (target: {target_count} modifications)")
            try:
                # Request more modifications from RAG to account for potential duplicates
                rag_smiles = self.rag_strategy.rag_modification_strategy(
                    self.parent.smiles,
                    property_gap,
                    target_count=target_count * 2  # Request 2x to ensure we have enough
                )
                modified_smiles.extend(rag_smiles)
                logger.info(f"RAG generated {len(rag_smiles)} modifications")
            except Exception as e:
                logger.warning(f"RAG strategy failed: {e}")
        
        # 3. Fill remaining slots with default strategy if needed
        remaining = target_count - len(modified_smiles)
        if remaining > 0:
            logger.info(f"Filling {remaining} remaining slots with default modification strategy")
            default_smiles = default_modification_strategy(
                self.parent.smiles,
                property_gap
            )
            
            # Add default modifications until we reach target_count
            for smiles in default_smiles:
                if smiles not in modified_smiles:
                    modified_smiles.append(smiles)
                    remaining -= 1
                    if remaining <= 0:
                        break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_smiles = []
        for smiles in modified_smiles:
            if smiles not in seen and smiles != self.parent.smiles:  # Don't include parent
                seen.add(smiles)
                unique_smiles.append(smiles)
                if len(unique_smiles) >= target_count:
                    break
        
        modified_smiles = unique_smiles
        
        if not modified_smiles:
            logger.warning(f"No modifications generated for {self.parent.smiles}")
            return []
        
        logger.info(f"Generated {len(modified_smiles)} unique candidate modifications (target: {target_count})")
        
        # 4. Evaluate all candidates
        candidates = []
        for smiles in modified_smiles:
            candidate = self.evaluate_candidate(smiles)
            if candidate is not None:
                candidates.append(candidate)
        
        logger.info(f"Successfully evaluated {len(candidates)} candidates (target: {target_count})")
        return candidates
    
    def evaluate_candidate(self, smiles: str) -> MoleculeState:
        """
        Evaluate a candidate molecule.
        
        Args:
            smiles: Candidate SMILES string
            
        Returns:
            MoleculeState object or None if invalid
        """
        try:
            # Skip multi-molecule SMILES (containing dots)
            if '.' in smiles:
                logger.debug(f"Skipping multi-molecule SMILES: {smiles}")
                return None
            
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.debug(f"Invalid SMILES string: {smiles}")
                return None
            
            # Canonicalize
            smiles = Chem.MolToSmiles(mol)
            
            # Predict properties
            predicted_props = predict_properties(smiles, self.config.system.models_directory)
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
                mae_weight=self.config.scoring.mae_weight,
                feasibility_weight=self.config.scoring.feasibility_weight,
                property_weights=self.config.scoring.property_weights,
                property_ranges=self.config.scoring.property_ranges
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
