"""
Main EnergeticDesigner class - high-level interface to the system.
"""

import os
import json
import logging
from typing import List
from data_structures import MoleculeState, PropertyTarget
from config import Config
from modules.initialization import load_dataset, find_closest_match
from orchestrator import BeamSearchEngine

logger = logging.getLogger(__name__)


class EnergeticDesigner:
    """
    High-level interface for energetic molecule design using beam search.
    """
    
    def __init__(self, target_properties: PropertyTarget, config: Config = None):
        """
        Initialize the designer.
        
        Args:
            target_properties: Target molecular properties
            config: System configuration (uses defaults if None)
        """
        self.target = target_properties
        self.config = config if config is not None else Config()
        
        self.dataset = None
        self.seed = None
        self.engine = None
        self.results = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.system.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def initialize(self):
        """
        Initialize the system: load dataset, find seed molecule.
        """
        logger.info("Initializing EnergeticDesigner")
        logger.info(f"Target properties: {self.target}")
        
        # Load dataset
        print(f"   📂 Loading dataset from: {self.config.system.dataset_path}")
        self.dataset = load_dataset(self.config.system.dataset_path)
        print(f"      ✓ Loaded {len(self.dataset)} molecules")
        
        # Find best seed molecule (uses MAPE distance)
        print(f"\n   🔍 Finding best seed molecule...")
        self.seed = find_closest_match(
            self.dataset,
            self.target
        )
        
        print(f"      ✓ Selected seed: {self.seed.smiles[:50]}{'...' if len(self.seed.smiles) > 50 else ''}")
        print(f"      ✓ Seed score: {self.seed.score:.4f}")
        
        logger.info(f"Initialization complete. Seed: {self.seed.smiles}")
    
    def run_design_loop(self) -> MoleculeState:
        """
        Execute the beam search optimization.
        
        Returns:
            Best molecule found
        """
        if self.seed is None:
            raise ValueError("Must call initialize() before run_design_loop()")
        
        # Create beam search engine
        self.engine = BeamSearchEngine(self.config, self.target)
        
        # Run beam search
        best_molecule = self.engine.run(self.seed)
        
        self.results = best_molecule
        return best_molecule
    
    def get_results(self) -> MoleculeState:
        """
        Get the final results.
        
        Returns:
            Best molecule found
        """
        if self.results is None:
            raise ValueError("Must call run_design_loop() first")
        
        return self.results
    
    def save_results(self, output_path: str):
        """
        Save results to file.
        
        Args:
            output_path: Path to save results (JSON or CSV)
        """
        if self.results is None:
            raise ValueError("No results to save")
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Prepare results data
        results_data = {
            'target_properties': self.target.to_dict(),
            'seed_molecule': self.seed.to_dict(),
            'best_molecule': self.results.to_dict(),
            'config': {
                'beam_width': self.config.beam_search.beam_width,
                'top_k': self.config.beam_search.top_k,
                'max_iterations': self.config.beam_search.max_iterations,
                'literature_enabled': self.config.literature.enable_literature_search
            }
        }
        
        # Save as JSON
        if output_path.endswith('.json'):
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        
        # Save as CSV
        elif output_path.endswith('.csv'):
            import pandas as pd
            df = pd.DataFrame([self.results.to_dict()])
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        else:
            raise ValueError("Output path must end with .json or .csv")
