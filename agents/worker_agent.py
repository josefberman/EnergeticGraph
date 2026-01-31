"""
Worker Agent (ChemistAgent) - Generates molecular variations.
"""

import logging
from typing import List, Dict, Optional, Tuple
from rdkit import Chem
from data_structures import MoleculeState, PropertyTarget
from modules.prediction import PropertyPredictor
from modules.feasibility import calculate_feasibility
from modules.scoring import calculate_total_score
from modules.strategy_pool import StrategyPoolModifier, default_modification_strategy
from modules.rag_retrieval import RAGPropertyRetriever, get_properties_with_rag, PaperCitation
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
        
        # Initialize RAG retriever if enabled
        self.rag_retriever = None
        if config.rag.enable_rag:
            self.rag_retriever = RAGPropertyRetriever(
                use_llm=config.rag.use_llm,
                max_papers=config.rag.max_papers,
                timeout=config.rag.timeout
            )
            logger.info("Initialized ChemistAgent with RAG property retrieval")
        
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
        
        Uses RAG to search for known property values in literature first,
        then falls back to ML prediction for any missing properties.
        
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
            
            # Get properties using RAG + ML fallback
            properties, sources, citations = self._get_properties_with_rag_fallback(smiles)
            
            if properties is None or not properties:
                logger.debug(f"Failed to get properties for {smiles}")
                return None
            
            # Log property sources and citations
            rag_count = sum(1 for s in sources.values() if 'literature' in s.lower())
            if rag_count > 0:
                logger.info(f"RAG retrieved {rag_count}/4 properties from literature for {smiles[:30]}...")
                # Display citations in CLI
                self._display_citations(citations, smiles)
            
            # Calculate feasibility (normalized SAScore)
            feasibility_score, is_feasible = calculate_feasibility(smiles)
            
            # Calculate total score
            target_dict = self.target.to_dict()
            score = calculate_total_score(
                properties,
                target_dict,
                feasibility_score,  # Normalized SAScore (0=feasible, 1=infeasible)
                mape_weight=self.config.scoring.mape_weight,
                sascore_weight=self.config.scoring.sascore_weight,
                property_weights=self.config.scoring.property_weights
            )
            
            # Build provenance string with property sources
            provenance_parts = [f"{self.parent.provenance} -> modification"]
            if rag_count > 0:
                provenance_parts.append(f"[RAG: {rag_count} props]")
            
            # Create MoleculeState
            candidate = MoleculeState(
                smiles=smiles,
                properties=properties,
                score=score,
                feasibility=feasibility_score,
                is_feasible=is_feasible,
                provenance=" ".join(provenance_parts),
                generation=self.parent.generation + 1,
                parent_smiles=self.parent.smiles
            )
            
            logger.debug(f"Evaluated {smiles}: score={score:.4f}, feasible={is_feasible}")
            return candidate
        
        except Exception as e:
            logger.error(f"Error evaluating candidate {smiles}: {e}")
            return None
    
    def _get_properties_with_rag_fallback(self, smiles: str) -> Tuple[Optional[Dict[str, float]], Dict[str, str], List[PaperCitation]]:
        """
        Get properties using RAG retrieval first, then ML prediction for missing values.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Tuple of (properties dict, sources dict, citations list)
        """
        properties = {}
        sources = {}
        citations = []
        
        required_props = ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid']
        
        # Step 1: Try RAG retrieval if enabled
        if self.rag_retriever is not None:
            try:
                rag_result = self.rag_retriever.retrieve_properties(smiles)
                
                for prop_name, prop_value in rag_result.properties.items():
                    if prop_value is not None:
                        properties[prop_name] = prop_value.value
                        sources[prop_name] = f"literature ({prop_value.source[:30]}...)" if len(prop_value.source) > 30 else f"literature ({prop_value.source})"
                
                # Collect citations
                citations = rag_result.citations or []
                        
            except Exception as e:
                logger.warning(f"RAG retrieval failed for {smiles[:30]}...: {e}")
        
        # Step 2: ML prediction for missing properties
        missing_props = [p for p in required_props if p not in properties]
        
        if missing_props:
            try:
                predicted = self.predictor.predict_properties(smiles)
                
                if predicted:
                    for prop_name in missing_props:
                        if prop_name in predicted and predicted[prop_name] is not None:
                            properties[prop_name] = predicted[prop_name]
                            sources[prop_name] = "predicted (XGBoost)"
            except Exception as e:
                logger.warning(f"ML prediction failed for {smiles[:30]}...: {e}")
        
        # Verify we have all required properties
        if not all(p in properties for p in required_props):
            return None, {}, []
        
        return properties, sources, citations
    
    def _display_citations(self, citations: List[PaperCitation], smiles: str):
        """
        Display RAG citations in CLI output.
        
        Args:
            citations: List of paper citations that provided property data
            smiles: SMILES string of the molecule
        """
        if not citations:
            return
        
        print(f"\n            📚 Literature References:")
        for i, citation in enumerate(citations, 1):
            # Format authors (first author et al. if more than 2)
            if len(citation.authors) == 0:
                author_str = "Unknown"
            elif len(citation.authors) == 1:
                author_str = citation.authors[0]
            elif len(citation.authors) == 2:
                author_str = f"{citation.authors[0]} & {citation.authors[1]}"
            else:
                author_str = f"{citation.authors[0]} et al."
            
            # Format properties found
            props_str = ", ".join(citation.properties_found) if citation.properties_found else "N/A"
            
            # Print citation in compact format
            title_short = citation.title[:55] + "..." if len(citation.title) > 55 else citation.title
            print(f"               [{i}] \"{title_short}\"")
            print(f"                   {author_str} | {citation.source_db}")
            if citation.properties_found:
                print(f"                   → Properties: {props_str}")