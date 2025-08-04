"""
Test script for Psi4 quantum chemical calculations
Demonstrates real quantum chemical calculations for explosives
"""

import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_psi4_basic():
    """Test basic Psi4 functionality"""
    print("Testing Psi4 basic functionality...")
    
    try:
        import psi4
        
        # Configure Psi4
        psi4.set_memory('500 MB')
        psi4.set_options({
            'basis': 'def2-TZVP',
            'scf_type': 'pk',
            'e_convergence': 1e-8,
            'd_convergence': 1e-8,
            'maxiter': 100
        })
        
        # Test with a simple molecule (nitromethane)
        mol_str = """
C 0.000000 0.000000 0.000000
N 1.500000 0.000000 0.000000
O 2.100000 0.000000 0.000000
O 1.800000 1.000000 0.000000
H -0.500000 0.000000 0.000000
H 0.000000 1.000000 0.000000
H 0.000000 0.000000 1.000000
"""
        
        mol = psi4.geometry(mol_str)
        print("✓ Psi4 molecule created successfully")
        
        # Perform DFT calculation
        energy, wfn = psi4.energy('B3LYP/def2-TZVP', molecule=mol, return_wfn=True)
        print(f"✓ DFT calculation successful: Energy = {energy:.6f} hartree")
        
        # Get molecular properties
        natom = mol.natom()
        charge = mol.molecular_charge()
        multiplicity = mol.multiplicity()
        
        print(f"✓ Molecular properties:")
        print(f"  - Number of atoms: {natom}")
        print(f"  - Charge: {charge}")
        print(f"  - Multiplicity: {multiplicity}")
        
        # Get geometry
        coords = mol.geometry().np
        symbols = [mol.symbol(i) for i in range(natom)]
        
        print(f"✓ Molecular geometry:")
        for i, (symbol, coord) in enumerate(zip(symbols, coords)):
            print(f"  {symbol}: {coord}")
        
        return True
        
    except Exception as e:
        print(f"✗ Psi4 test failed: {e}")
        return False

def test_rdkit_integration():
    """Test RDKit integration with Psi4"""
    print("\nTesting RDKit integration...")
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import psi4
        
        # Create molecule from SMILES
        smiles = "C[N+](=O)[O-]"  # Nitromethane
        rdmol = Chem.MolFromSmiles(smiles)
        
        if rdmol is None:
            print("✗ Failed to parse SMILES")
            return False
        
        # Generate 3D coordinates
        AllChem.EmbedMolecule(rdmol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(rdmol)
        
        # Convert to Psi4 format
        conf = rdmol.GetConformer()
        mol_str = ""
        
        for atom in rdmol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            mol_str += f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n"
        
        # Create Psi4 molecule
        mol = psi4.geometry(mol_str)
        print("✓ RDKit to Psi4 conversion successful")
        
        # Perform calculation
        energy, wfn = psi4.energy('B3LYP/def2-TZVP', molecule=mol, return_wfn=True)
        print(f"✓ Calculation successful: Energy = {energy:.6f} hartree")
        
        return True
        
    except Exception as e:
        print(f"✗ RDKit integration test failed: {e}")
        return False

def test_explosive_calculations():
    """Test calculations for explosive molecules"""
    print("\nTesting explosive molecule calculations...")
    
    try:
        import psi4
        
        # Test TNT molecule
        tnt_smiles = "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]"
        
        # Simplified TNT geometry
        tnt_geometry = """
C 0.000000 0.000000 0.000000
C 1.400000 0.000000 0.000000
C 2.100000 1.200000 0.000000
C 1.400000 2.400000 0.000000
C 0.000000 2.400000 0.000000
C -0.700000 1.200000 0.000000
N 3.200000 1.200000 0.000000
O 3.900000 0.300000 0.000000
O 3.900000 2.100000 0.000000
N -1.900000 1.200000 0.000000
O -2.600000 0.300000 0.000000
O -2.600000 2.100000 0.000000
H 0.000000 -0.500000 0.000000
H -0.700000 3.000000 0.000000
H 1.400000 3.000000 0.000000
"""
        
        mol = psi4.geometry(tnt_geometry)
        
        # Perform DFT calculation
        energy, wfn = psi4.energy('B3LYP/def2-TZVP', molecule=mol, return_wfn=True)
        print(f"✓ TNT calculation successful: Energy = {energy:.6f} hartree")
        
        # Calculate molecular properties
        natom = mol.natom()
        symbols = [mol.symbol(i) for i in range(natom)]
        
        # Count atoms for oxygen balance
        c_count = symbols.count('C')
        h_count = symbols.count('H')
        n_count = symbols.count('N')
        o_count = symbols.count('O')
        
        print(f"✓ TNT molecular composition:")
        print(f"  - C: {c_count}, H: {h_count}, N: {n_count}, O: {o_count}")
        
        # Calculate oxygen balance
        molecular_mass = c_count * 12.011 + h_count * 1.008 + n_count * 14.007 + o_count * 15.999
        oxygen_balance = (o_count - 2*c_count - h_count/2) * 16 / molecular_mass * 100
        
        print(f"  - Molecular mass: {molecular_mass:.2f} g/mol")
        print(f"  - Oxygen balance: {oxygen_balance:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Explosive calculations test failed: {e}")
        return False

def main():
    """Run all Psi4 tests"""
    print("=" * 60)
    print("PSI4 QUANTUM CHEMICAL CALCULATIONS - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic Psi4 Functionality", test_psi4_basic),
        ("RDKit Integration", test_rdkit_integration),
        ("Explosive Molecule Calculations", test_explosive_calculations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
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
        print("🎉 All Psi4 tests passed! Quantum chemical calculations ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 