"""
Main entry point for the Beam Search Molecular Design System.
"""

import argparse
import os
from dotenv import load_dotenv

# Suppress RDKit warnings (must be done before importing other modules that use RDKit)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data_structures import PropertyTarget
from config import Config
from designer import EnergeticDesigner

# Load .env file
load_dotenv()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Beam Search Molecular Design System for Energetic Materials'
    )
    
    # Target properties
    parser.add_argument('--density', type=float, required=True, 
                       help='Target density (g/cm³)')
    parser.add_argument('--velocity', type=float, required=True,
                       help='Target detonation velocity (m/s)')
    parser.add_argument('--pressure', type=float, required=True,
                       help='Target detonation pressure (GPa)')
    parser.add_argument('--hf', type=float, required=True,
                       help='Target formation enthalpy (kJ/mol)')
    
    # Configuration
    parser.add_argument('--beam-width', type=int, default=10,
                       help='Beam width (default: 10)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Top K candidates to keep (default: 5)')
    parser.add_argument('--max-iter', type=int, default=20,
                       help='Maximum iterations (default: 20)')
    
    # RAG configuration
    parser.add_argument('--disable-rag', action='store_true',
                       help='Disable RAG-based modification strategy (enabled by default)')
    parser.add_argument('--openai-key', type=str, default=None,
                       help='OpenAI API key (if not set in env)')
    parser.add_argument('--arxiv-max', type=int, default=3,
                       help='Max Arxiv papers to retrieve (default: 3)')
    
    # System
    parser.add_argument('--dataset', type=str, default='./sample_start_molecules.csv',
                       help='Path to dataset file')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Path to models directory')
    parser.add_argument('--output', type=str, default='./output/results.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Create target properties
    target = PropertyTarget(
        density=args.density,
        det_velocity=args.velocity,
        det_pressure=args.pressure,
        hf_solid=args.hf
    )
    
    # Create configuration
    config = Config()
    config.beam_search.beam_width = args.beam_width
    config.beam_search.top_k = args.top_k
    config.beam_search.max_iterations = args.max_iter
    
    # Disable RAG if flag is provided
    if args.disable_rag:
        config.rag.enable_rag = False
    
    # Override API key if provided via command line
    if args.openai_key:
        config.rag.openai_api_key = args.openai_key
    # Otherwise, it will use the value from .env (auto-loaded in config.py)
    
    config.rag.arxiv_max_results = args.arxiv_max
    
    config.system.dataset_path = args.dataset
    config.system.models_directory = args.models_dir
    config.system.output_directory = os.path.dirname(args.output)
    
    # Create designer
    designer = EnergeticDesigner(target, config)
    
    # Initialize
    print("Initializing system...")
    designer.initialize()
    
    # Run design loop
    print("\nRunning beam search optimization...")
    print(f"Target: {target}")
    print(f"RAG enabled: {config.rag.enable_rag}")
    print(f"Beam width: {config.beam_search.beam_width}")
    print(f"Top K: {config.beam_search.top_k}")
    print(f"Max iterations: {config.beam_search.max_iterations}")
    print()
    
    best_molecule = designer.run_design_loop()
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Best SMILES: {best_molecule.smiles}")
    print(f"Score: {best_molecule.score:.4f}")
    print(f"Feasibility: {best_molecule.feasibility:.2f}")
    print(f"Properties:")
    for prop, value in best_molecule.properties.items():
        target_val = target.to_dict()[prop]
        print(f"  {prop}: {value:.2f} (target: {target_val:.2f})")
    print("="*60)
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    designer.save_results(args.output)
    print("Done!")


if __name__ == '__main__':
    main()
