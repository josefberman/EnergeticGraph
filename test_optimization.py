#!/usr/bin/env python3
"""
Test script to verify the molecular optimization process works without hanging
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

def test_optimization():
    """Test the optimization process"""
    print("Testing molecular optimization process...")
    
    try:
        from molecular_optimizer_agent import MolecularOptimizationAgent
        
        # Initialize agent with smaller parameters for testing
        agent = MolecularOptimizationAgent(beam_width=3, max_iterations=2, convergence_threshold=0.01)
        
        # Test with a simple CSV
        test_csv = "sample_optimization_input.csv"
        
        print(f"Running optimization with {test_csv}...")
        results = agent.process_csv_input(test_csv, verbose=True)
        
        if "error" in results:
            print(f"❌ Optimization failed: {results['error']}")
            return False
        else:
            print("✅ Optimization completed successfully!")
            print(f"Best molecule: {results.get('best_molecule', 'N/A')}")
            print(f"Best score: {results.get('best_score', 'N/A')}")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_modification_generation():
    """Test modification generation without hanging"""
    print("Testing modification generation...")
    
    try:
        from molecular_optimizer_agent import MolecularOptimizationAgent
        
        agent = MolecularOptimizationAgent()
        
        # Test with a simple molecule
        test_smiles = "CC1=CC=C(C=C1)[N+](=O)[O-]"  # TNT
        target_properties = {"Density": 1.8, "Detonation velocity": 8000.0}
        
        print(f"Generating modifications for {test_smiles}...")
        modifications = agent._get_modifications_with_agent(test_smiles, target_properties, 0.5, verbose=True)
        
        print(f"✅ Generated {len(modifications)} modifications")
        return True
        
    except Exception as e:
        print(f"❌ Modification generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Molecular Optimization Test")
    print("=" * 50)
    
    # Test modification generation first
    mod_success = test_modification_generation()
    
    if mod_success:
        # Test full optimization
        opt_success = test_optimization()
        
        print("\n" + "=" * 50)
        print("Test Results")
        print("=" * 50)
        
        if opt_success:
            print("🎉 All tests passed! The optimization system is working correctly.")
        else:
            print("⚠️  Optimization test failed, but modification generation works.")
    else:
        print("❌ Modification generation failed. Check the system setup.")

if __name__ == "__main__":
    main() 