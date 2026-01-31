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
        
        # Print seed info
        print()
        print(f"   🌱 Seed Molecule: {seed_molecule.smiles[:50]}{'...' if len(seed_molecule.smiles) > 50 else ''}")
        print(f"      Initial Score: {seed_molecule.score:.4f}")
        print()
        
        logger.info(f"Starting beam search with seed: {seed_molecule.smiles}")
        
        for iteration in range(self.beam_config.max_iterations):
            # Print iteration header
            print()
            print(f"┌{'─' * 58}┐")
            print(f"│  📍 ITERATION {iteration + 1}/{self.beam_config.max_iterations}" + " " * (43 - len(str(iteration + 1)) - len(str(self.beam_config.max_iterations))) + "│")
            print(f"└{'─' * 58}┘")
            
            # Generate candidates from current beam
            all_candidates = []
            
            for idx, parent_mol in enumerate(current_beam):
                print(f"\n   🔬 Processing Parent {idx + 1}/{len(current_beam)}:")
                print(f"      SMILES: {parent_mol.smiles[:45]}{'...' if len(parent_mol.smiles) > 45 else ''}")
                print(f"      Score:  {parent_mol.score:.4f}")
                print()
                
                # Create worker agent for this parent
                agent = ChemistAgent(parent_mol, self.target, self.config)
                
                # Generate variations
                new_candidates = agent.generate_variations()
                all_candidates.extend(new_candidates)
                
                print(f"\n      ✓ Generated {len(new_candidates)} candidate variations")
            
            # Stats
            print(f"\n   📈 Iteration Statistics:")
            print(f"      • Total candidates:    {len(all_candidates)}")
            
            # Filter feasible candidates
            feasible_candidates = [m for m in all_candidates if m.is_feasible]
            print(f"      • Feasible candidates: {len(feasible_candidates)}")
            
            if not feasible_candidates:
                print(f"\n   ⚠️  No feasible candidates found. Stopping search.")
                break
            
            # Remove duplicates (by SMILES)
            unique_candidates = self._remove_duplicates(feasible_candidates)
            print(f"      • Unique candidates:   {len(unique_candidates)}")
            
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
                print(f"\n   🌟 NEW BEST FOUND!")
                print(f"      SMILES: {self.best_ever.smiles[:45]}{'...' if len(self.best_ever.smiles) > 45 else ''}")
                print(f"      MAPE:   {best_mape:.2f}%")
            
            # Check convergence
            if iteration > 0:
                prev_best_score = current_beam[0].score
                curr_best_score = next_beam[0].score
                improvement = prev_best_score - curr_best_score
                
                print(f"\n   📉 Score Improvement: {improvement:.6f}")
                
                if improvement < self.beam_config.convergence_threshold:
                    print(f"\n   ✅ Converged! (improvement < {self.beam_config.convergence_threshold})")
                    break
            
            # Update beam
            current_beam = next_beam
            self.history.append(current_beam)
        
        # Final summary
        print()
        print(f"┌{'─' * 58}┐")
        print(f"│  ✅ BEAM SEARCH COMPLETE" + " " * 33 + "│")
        print(f"└{'─' * 58}┘")
        print()
        
        logger.info(f"Beam search complete. Best: {self.best_ever.smiles}")
        
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
        print(f"\n   🏅 Top {min(3, len(beam))} Candidates This Iteration:")
        print()
        
        for i, mol in enumerate(beam[:3]):  # Show top 3
            mape = self.calculate_mape(mol)
            feasibility_pct = (1 - mol.feasibility) * 100
            
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
            
            print(f"      {medal} Rank {i+1}:")
            print(f"         SMILES:      {mol.smiles[:40]}{'...' if len(mol.smiles) > 40 else ''}")
            print(f"         Score:       {mol.score:.4f}")
            print(f"         MAPE:        {mape:.1f}%")
            print(f"         Feasibility: {feasibility_pct:.0f}%")
            print()
        
        logger.debug(f"Iteration {iteration}: beam size = {len(beam)}")
