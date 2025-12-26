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
        
        # 2. EXHAUSTIVE generation loop - keep trying until we reach beam_width
        candidates = []
        seen = set()  # Track already-seen SMILES to avoid duplicates
        max_attempts = 10  # Increased for exhaustive search
        attempt = 0
        
        logger.info(f"Starting EXHAUSTIVE candidate generation for beam_width={target_count}")
        
        while len(candidates) < target_count and attempt < max_attempts:
            attempt += 1
            needed = target_count - len(candidates)
            logger.info(f"Exhaustive attempt {attempt}/{max_attempts}: Need {needed} more (have {len(candidates)})")
            
            # Request much more than we need
            request_count = max(needed * 5, target_count * 3)  
            batch_smiles = []
            
            # STRATEGY 1: RAG with multiple queries (if enabled)
            if self.config.rag.enable_rag and self.rag_strategy is not None:
                try:
                    rag_smiles = self.rag_strategy.rag_modification_strategy(
                        self.parent.smiles,
                        property_gap,
                        target_count=request_count
                    )
                    batch_smiles.extend(rag_smiles)
                    logger.info(f"RAG returned {len(rag_smiles)} modifications")
                except Exception as e:
                    logger.warning(f"RAG strategy failed: {e}")

            # STRATEGY 2: ALWAYS run default strategy
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
            
            # STRATEGY 3: Direct diverse modifications
            from modules.modification_tools import generate_diverse_modifications, apply_all_modifications
            try:
                diverse_smiles = generate_diverse_modifications(self.parent.smiles, target_count=request_count)
                batch_smiles.extend(diverse_smiles)
                logger.info(f"Diverse generation added {len(diverse_smiles)} modifications")
            except Exception as e:
                logger.warning(f"Diverse generation failed: {e}")
            
            # Filter to unique unseen SMILES
            unique_batch = []
            for s in batch_smiles:
                if s not in seen and s != self.parent.smiles:
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
            
            # STRATEGY 4: Recursive expansion from existing candidates
            if len(candidates) < target_count and len(candidates) > 0:
                logger.info(f"Recursive expansion from {len(candidates)} existing candidates")
                
                # Expand from ALL existing candidates, not just top 5
                for existing in candidates:
                    if len(candidates) >= target_count:
                        break
                    try:
                        # Level 1 expansion
                        expanded = apply_all_modifications(existing.smiles)
                        for exp_smiles in expanded:
                            if exp_smiles not in seen and exp_smiles != self.parent.smiles:
                                seen.add(exp_smiles)
                                exp_candidate = self.evaluate_candidate(exp_smiles)
                                if exp_candidate is not None:
                                    candidates.append(exp_candidate)
                                    if len(candidates) >= target_count:
                                        break
                    except Exception as e:
                        logger.debug(f"Expansion failed: {e}")
            
            # STRATEGY 5: Deep recursive expansion (depth-2) if still not enough
            if len(candidates) < target_count and attempt >= 3:
                logger.info(f"Deep expansion (depth-2) - still need {target_count - len(candidates)} more")
                for existing in candidates[:min(3, len(candidates))]:
                    if len(candidates) >= target_count:
                        break
                    try:
                        level1 = apply_all_modifications(existing.smiles)
                        for l1_smiles in level1[:3]:
                            if len(candidates) >= target_count:
                                break
                            level2 = apply_all_modifications(l1_smiles)
                            for l2_smiles in level2[:5]:
                                if l2_smiles not in seen:
                                    seen.add(l2_smiles)
                                    candidate = self.evaluate_candidate(l2_smiles)
                                    if candidate is not None:
                                        candidates.append(candidate)
                                        if len(candidates) >= target_count:
                                            break
                    except Exception as e:
                        logger.debug(f"Deep expansion failed: {e}")
        
        if not candidates:
            logger.warning(f"No valid candidates generated for {self.parent.smiles} after {attempt} attempts")
            return []
        
        # Return exactly target_count candidates (truncate if we have more)
        final_candidates = candidates[:target_count]
        logger.info(f"EXHAUSTIVE SEARCH: Returning {len(final_candidates)} candidates (target: {target_count})")
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
