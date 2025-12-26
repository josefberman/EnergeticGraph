"""
Beam Search Orchestrator - manages the beam search algorithm.
"""

import logging
from typing import List
from data_structures import MoleculeState, PropertyTarget
from agents.worker_agent import ChemistAgent
from config import Config

logger = logging.getLogger(__name__)


class BeamSearchEngine:
    """
    Orchestrates the beam search optimization process.
    """
    
    def __init__(self, config: Config, target_properties: PropertyTarget):
        """
        Initialize beam search engine.
        
        Args:
            config: System configuration
            target_properties: Target molecular properties
        """
        self.config = config
        self.target = target_properties
        self.beam_config = config.beam_search
        
        # History tracking
        self.history = []
        self.best_ever = None
    
    def calculate_mape(self, molecule: MoleculeState) -> float:
        """
        Calculate Mean Absolute Percentage Error relative to target values.
        
        Args:
            molecule: Molecule to evaluate
            
        Returns:
            MAPE as percentage (lower is better)
        """
        target_dict = self.target.to_dict()
        props = molecule.properties
        
        errors = []
        for key in ['Density', 'Det Velocity', 'Det Pressure', 'Hf solid']:
            if key in target_dict and key in props:
                target_val = abs(target_dict[key])
                if target_val > 0:
                    error_pct = abs(props[key] - target_dict[key]) / target_val * 100
                    errors.append(error_pct)
        
        return sum(errors) / len(errors) if errors else 100.0
    
    def run(self, seed_molecule: MoleculeState) -> MoleculeState:
        """
        Run beam search algorithm.
        
        Args:
            seed_molecule: Initial seed molecule
            
        Returns:
            Best molecule found
        """
        current_beam = [seed_molecule]
        self.best_ever = seed_molecule
        
        logger.info(f"Starting beam search with seed: {seed_molecule.smiles}")
        logger.info(f"Seed score: {seed_molecule.score:.4f}")
        
        for iteration in range(self.beam_config.max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{self.beam_config.max_iterations}")
            logger.info(f"{'='*60}")
            
            # Generate candidates from current beam
            all_candidates = []
            
            for parent_mol in current_beam:
                logger.info(f"Processing parent: {parent_mol.smiles} (score: {parent_mol.score:.4f})")
                
                # Create worker agent for this parent
                agent = ChemistAgent(parent_mol, self.target, self.config)
                
                # Generate variations
                new_candidates = agent.generate_variations()
                all_candidates.extend(new_candidates)
                
                logger.info(f"  Generated {len(new_candidates)} new candidates")
            
            logger.info(f"Total candidates this iteration: {len(all_candidates)}")
            
            # Filter feasible candidates
            feasible_candidates = [m for m in all_candidates if m.is_feasible]
            logger.info(f"Feasible candidates: {len(feasible_candidates)}")
            
            if not feasible_candidates:
                logger.warning("No feasible candidates found. Stopping.")
                break
            
            # Remove duplicates (by SMILES)
            unique_candidates = self._remove_duplicates(feasible_candidates)
            logger.info(f"Unique candidates: {len(unique_candidates)}")
            
            # Rank by MAPE (lower is better) instead of combined score
            ranked_candidates = sorted(unique_candidates, key=lambda x: self.calculate_mape(x))
            
            # Prune to top_k
            next_beam = ranked_candidates[:self.beam_config.top_k]
            
            # Log iteration results
            self.log_iteration(iteration + 1, next_beam)
            
            # Update best ever by MAPE comparison
            best_mape = self.calculate_mape(next_beam[0])
            best_ever_mape = self.calculate_mape(self.best_ever)
            if best_mape < best_ever_mape:
                self.best_ever = next_beam[0]
                logger.info(f"*** NEW BEST: {self.best_ever.smiles} (MAPE: {best_mape:.2f}%) ***")
            
            # Check convergence
            if iteration > 0:
                prev_best_score = current_beam[0].score
                curr_best_score = next_beam[0].score
                improvement = prev_best_score - curr_best_score
                
                logger.info(f"Improvement: {improvement:.6f}")
                
                if improvement < self.beam_config.convergence_threshold:
                    logger.info(f"Converged (improvement < {self.beam_config.convergence_threshold})")
                    break
            
            # Update beam
            current_beam = next_beam
            self.history.append(current_beam)
        
        logger.info(f"\n{'='*60}")
        logger.info("Beam search complete")
        logger.info(f"Best molecule: {self.best_ever.smiles}")
        logger.info(f"Best score: {self.best_ever.score:.4f}")
        logger.info(f"Properties: {self.best_ever.properties}")
        logger.info(f"Feasibility: {self.best_ever.feasibility:.2f}")
        logger.info(f"{'='*60}\n")
        
        return self.best_ever
    
    def _remove_duplicates(self, molecules: List[MoleculeState]) -> List[MoleculeState]:
        """Remove duplicate molecules by SMILES."""
        seen = set()
        unique = []
        
        for mol in molecules:
            if mol.smiles not in seen:
                seen.add(mol.smiles)
                unique.append(mol)
        
        return unique
    
    def log_iteration(self, iteration: int, beam: List[MoleculeState]):
        """
        Log iteration results.
        
        Args:
            iteration: Iteration number
            beam: Current beam
        """
        logger.info(f"\nIteration {iteration} Results:")
        logger.info(f"Beam size: {len(beam)}")
        
        for i, mol in enumerate(beam[:3]):  # Show top 3
            logger.info(f"  {i+1}. SMILES: {mol.smiles}")
            logger.info(f"     Score: {mol.score:.4f}")
            logger.info(f"     Feasibility: {mol.feasibility:.2f}")
            logger.info(f"     Properties: {mol.properties}")
