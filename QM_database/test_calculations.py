"""
Test script for explosives quantum chemical calculations
Verifies that all components work correctly before full database generation
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explosives_list import get_explosives_list
from quantum_calculator import ExplosiveQuantumCalculator
from advanced_calculations import AdvancedExplosiveCalculator

def test_basic_calculations():
    """Test basic quantum chemical calculations"""
    print("Testing basic quantum chemical calculations...")
    
    # Initialize calculator
    calc = ExplosiveQuantumCalculator()
    
    # Test with a simple molecule (TNT)
    test_smiles = "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]"
    test_name = "TNT_Test"
    
    try:
        results = calc.calculate_molecular_properties(test_smiles, test_name)
        
        if results:
            print("✓ Basic calculations successful")
            print(f"  - Density: {results.get('density_g_cm3', 'N/A')} g/cm³")
            print(f"  - Heat of Formation: {results.get('heat_of_formation_kcal_mol', 'N/A')} kcal/mol")
            print(f"  - Oxygen Balance: {results.get('oxygen_balance_percent', 'N/A')}%")
            return True
        else:
            print("✗ Basic calculations failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic calculations error: {e}")
        return False

def test_advanced_calculations():
    """Test advanced quantum chemical calculations"""
    print("\nTesting advanced quantum chemical calculations...")
    
    # Initialize calculators
    quantum_calc = ExplosiveQuantumCalculator()
    advanced_calc = AdvancedExplosiveCalculator()
    
    # Test with RDX
    test_smiles = "C1N2CN3CN1CN2CN3"
    test_name = "RDX_Test"
    
    try:
        # Convert SMILES to PySCF molecule
        mol = quantum_calc._smiles_to_pyscf(test_smiles, test_name)
        
        if mol is None:
            print("✗ SMILES to PySCF conversion failed")
            return False
        
        # Test phonon calculations
        phonon_results = advanced_calc.calculate_phonon_dispersion(mol)
        print("✓ Phonon dispersion calculation successful")
        
        # Test EOS calculations
        eos_results = advanced_calc.calculate_equation_of_state(mol)
        print("✓ Equation of state calculation successful")
        
        # Test elastic moduli
        elastic_results = advanced_calc.calculate_elastic_moduli(mol)
        print("✓ Elastic moduli calculation successful")
        
        # Test cohesive energy
        cohesive_results = advanced_calc.calculate_cohesive_energy(mol)
        print("✓ Cohesive energy calculation successful")
        
        # Test trigger bond analysis
        trigger_results = advanced_calc.calculate_trigger_bond_analysis(mol)
        print("✓ Trigger bond analysis successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced calculations error: {e}")
        return False

def test_database_generation():
    """Test database generation with a small subset"""
    print("\nTesting database generation...")
    
    # Get first 3 explosives for testing
    explosives_list = get_explosives_list()[:3]
    
    try:
        # Initialize calculator
        calc = ExplosiveQuantumCalculator()
        
        results = []
        for explosive in explosives_list:
            print(f"  Processing {explosive['name']}...")
            
            result = calc.calculate_molecular_properties(
                explosive['smiles'], 
                explosive['name']
            )
            
            if result:
                result.update({
                    'category': explosive['category'],
                    'formula': explosive['formula']
                })
                results.append(result)
                print(f"    ✓ {explosive['name']} completed")
            else:
                print(f"    ✗ {explosive['name']} failed")
        
        if results:
            # Save test results
            df = pd.DataFrame(results)
            test_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(test_filename, index=False)
            print(f"✓ Test database generation successful - saved to {test_filename}")
            return True
        else:
            print("✗ Test database generation failed - no results")
            return False
            
    except Exception as e:
        print(f"✗ Test database generation error: {e}")
        return False

def test_property_calculations():
    """Test specific property calculations"""
    print("\nTesting specific property calculations...")
    
    # Initialize calculator
    calc = ExplosiveQuantumCalculator()
    
    # Test oxygen balance calculation
    test_molecules = [
        {"name": "TNT", "smiles": "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]", "expected_ob": -74.0},
        {"name": "PETN", "smiles": "C(C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "expected_ob": -10.1},
        {"name": "RDX", "smiles": "C1N2CN3CN1CN2CN3", "expected_ob": -21.6}
    ]
    
    for molecule in test_molecules:
        try:
            mol = calc._smiles_to_pyscf(molecule['smiles'], molecule['name'])
            if mol:
                ob = calc._calculate_oxygen_balance(mol)
                print(f"  {molecule['name']}: OB = {ob:.1f}% (expected ~{molecule['expected_ob']}%)")
            else:
                print(f"  ✗ {molecule['name']}: SMILES conversion failed")
        except Exception as e:
            print(f"  ✗ {molecule['name']}: Error - {e}")

def test_memory_usage():
    """Test memory usage and performance"""
    print("\nTesting memory usage and performance...")
    
    import psutil
    import time
    
    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Test with a simple calculation
    calc = ExplosiveQuantumCalculator(max_memory=1000)  # 1GB limit
    
    start_time = time.time()
    
    try:
        result = calc.calculate_molecular_properties(
            "C[N+](=O)[O-]",  # Nitromethane
            "Nitromethane_Test"
        )
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Calculation time: {end_time - start_time:.2f} seconds")
        print(f"Final memory usage: {final_memory:.1f} MB")
        print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
        
        if result:
            print("✓ Memory test successful")
            return True
        else:
            print("✗ Memory test failed - no results")
            return False
            
    except Exception as e:
        print(f"✗ Memory test error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("EXPLOSIVES QUANTUM CHEMICAL DATABASE - TEST SUITE")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Basic Calculations", test_basic_calculations),
        ("Advanced Calculations", test_advanced_calculations),
        ("Database Generation", test_database_generation),
        ("Property Calculations", test_property_calculations),
        ("Memory Usage", test_memory_usage)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database generation ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 