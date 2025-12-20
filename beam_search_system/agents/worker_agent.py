"""
Worker Agent (ChemistAgent) - Generates molecular variations.
"""

import logging
from typing import List, Dict
from ..data_structures import MoleculeState, PropertyTarget
from ..modules.prediction import PropertyPredictor
from ..modules.feasibility import calculate_feasibility
from ..modules.scoring import calculate_total_score
from ..modules.rag_strategy import RAGModificationStrategy, default_modification_strategy
from ..config import Config

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
        2. Try RAG strategy if enabled
        3. Fall back to default if RAG fails or disabled
        4. Evaluate all candidates
        5. Return MoleculeState objects
        
        Returns:
            List of candidate MoleculeState objects
        """
        # 1. Analyze property gap
        property_gap = self.analyze_property_gap()
        
        # 2. Generate modifications
        modified_smiles = []
        
        if self.config.rag.enable_rag and self.rag_strategy is not None:
            logger.info("Attempting RAG-based modification strategy")
            try:
                modified_smiles = self.rag_strategy.rag_modification_strategy(
                    self.parent.smiles,
                    property_gap
                )
            except Exception as e:
                logger.warning(f"RAG strategy failed: {e}")
        
        # 3. Fall back to default strategy if needed
        if not modified_smiles:
            logger.info("Using default modification strategy")
            modified_smiles = default_modification_strategy(
                self.parent.smiles,
                property_gap
            )
        
        if not modified_smiles:
            logger.warning(f"No modifications generated for {self.parent.smiles}")
            return []
        
        logger.info(f"Generated {len(modified_smiles)} candidate modifications")
        
        # 4. Evaluate all candidates
        candidates = []
        for smiles in modified_smiles:
            candidate = self.evaluate_candidate(smiles)
            if candidate is not None:
                candidates.append(candidate)
        
        return candidates
    
    def evaluate_candidate(self, smiles: str) -> MoleculeState:
        """
        Evaluate a candidate molecule: predict properties, check feasibility, calculate score.
        
        Args:
            smiles: Candidate SMILES
            
        Returns:
            MoleculeState object, or None if evaluation fails
        """
        try:
            # Predict properties
            predicted_props = self.predictor.predict_properties(smiles)
            if predicted_props is None:
                logger.debug(f"Failed to predict properties for {smiles}")
                return None
            
            # Check feasibility
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
