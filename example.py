"""
Simple example script to test the beam search system.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_structures import PropertyTarget
from config import Config
from designer import EnergeticDesigner


def main():
    """Run a simple test of the system."""
    
    print("="*60)
    print("Beam Search Molecular Design System - Test Example")
    print("="*60)
    
    # Define target properties (typical energetic material)
    target = PropertyTarget(
        density=1.85,          # g/cm³
        det_velocity=8500.0,   # m/s
        det_pressure=35.0,     # GPa
        hf_solid=-50.0         # kJ/mol
    )
    
    print(f"\nTarget Properties:")
    print(f"  Density: {target.density} g/cm³")
    print(f"  Detonation Velocity: {target.det_velocity} m/s")
    print(f"  Detonation Pressure: {target.det_pressure} GPa")
    print(f"  Formation Enthalpy: {target.hf_solid} kJ/mol")
    
    # Create configuration
    config = Config()
    
    # Adjust for faster testing
    config.beam_search.beam_width = 5
    config.beam_search.top_k = 3
    config.beam_search.max_iterations = 5
    
    # Disable RAG for this simple test (or enable if you have API key)
    config.rag.enable_rag = False
    
    # Check for OpenAI API key
    if 'OPENAI_API_KEY' in os.environ:
        print("\n[INFO] OpenAI API key found. RAG can be enabled.")
        use_rag = input("Enable RAG? (y/n): ").strip().lower() == 'y'
        config.rag.enable_rag = use_rag
    else:
        print("\n[INFO] No OpenAI API key found. RAG disabled.")
        print("[INFO] Set OPENAI_API_KEY environment variable to enable RAG.")
    
    print(f"\nConfiguration:")
    print(f"  Beam Width: {config.beam_search.beam_width}")
    print(f"  Top K: {config.beam_search.top_k}")
    print(f"  Max Iterations: {config.beam_search.max_iterations}")
    print(f"  RAG Enabled: {config.rag.enable_rag}")
    
    # Create designer
    print("\nInitializing EnergeticDesigner...")
    designer = EnergeticDesigner(target, config)
    
    try:
        # Initialize system
        print("Loading dataset and finding seed molecule...")
        designer.initialize()
        
        print(f"\nSeed molecule: {designer.seed.smiles}")
        print(f"Seed score: {designer.seed.score:.4f}")
        print(f"Seed properties: {designer.seed.properties}")
        
        # Run optimization
        print("\n" + "="*60)
        print("Starting Beam Search Optimization")
        print("="*60)
        
        best_molecule = designer.run_design_loop()
        
        # Display results
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"\nBest SMILES: {best_molecule.smiles}")
        print(f"Final Score: {best_molecule.score:.4f}")
        print(f"Feasibility: {best_molecule.feasibility:.2f}")
        print(f"Generation: {best_molecule.generation}")
        
        print(f"\nPredicted Properties:")
        for prop, value in best_molecule.properties.items():
            target_val = target.to_dict()[prop]
            diff = value - target_val
            print(f"  {prop}: {value:.2f} (target: {target_val:.2f}, diff: {diff:+.2f})")
        
        # Save results
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "example_results.json")
        
        print(f"\nSaving results to {output_path}...")
        designer.save_results(output_path)
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
    
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
