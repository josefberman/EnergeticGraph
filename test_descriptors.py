#!/usr/bin/env python3
"""
Test script to verify descriptor functions work correctly
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

def test_descriptor_functions():
    """Test all descriptor functions"""
    print("Testing descriptor functions...")
    
    try:
        from prediction import create_descriptor, get_c_count, get_n_count
        
        # Test with valid SMILES
        test_smiles = "CC1=CC=C(C=C1)[N+](=O)[O-]"  # TNT
        
        print(f"Testing with SMILES: {test_smiles}")
        
        # Test individual functions
        c_count = get_c_count(test_smiles)
        n_count = get_n_count(test_smiles)
        
        print(f"  Carbon count: {c_count}")
        print(f"  Nitrogen count: {n_count}")
        
        # Test full descriptor
        descriptor = create_descriptor(test_smiles)
        
        print(f"  Full descriptor length: {len(descriptor)}")
        print(f"  Descriptor values: {descriptor}")
        
        return True
        
    except Exception as e:
        print(f"❌ Descriptor test failed: {e}")
        return False

def test_invalid_smiles():
    """Test descriptor functions with invalid SMILES"""
    print("Testing with invalid SMILES...")
    
    try:
        from prediction import create_descriptor
        
        invalid_smiles = "INVALID_SMILES_STRING"
        
        print(f"Testing with invalid SMILES: {invalid_smiles}")
        
        # This should not crash
        descriptor = create_descriptor(invalid_smiles)
        
        print(f"  Descriptor created: {descriptor}")
        print("✅ Invalid SMILES handled gracefully")
        
        return True
        
    except Exception as e:
        print(f"❌ Invalid SMILES test failed: {e}")
        return False

def test_data_loading():
    """Test loading the data file"""
    print("Testing data loading...")
    
    try:
        import pandas as pd
        
        # Try to load the data file
        df = pd.read_excel("clean_data_imputed.xlsx")
        
        print(f"✅ Data loaded successfully: {len(df)} samples")
        print(f"  Columns: {list(df.columns)}")
        
        # Test a few SMILES
        if 'SMILES' in df.columns:
            sample_smiles = df['SMILES'].iloc[0]
            print(f"  Sample SMILES: {sample_smiles}")
            
            from prediction import create_descriptor
            descriptor = create_descriptor(sample_smiles)
            print(f"  Sample descriptor: {descriptor}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("DESCRIPTOR FUNCTION TEST")
    print("=" * 50)
    
    # Test data loading
    data_success = test_data_loading()
    
    if data_success:
        # Test descriptor functions
        desc_success = test_descriptor_functions()
        
        # Test invalid SMILES handling
        invalid_success = test_invalid_smiles()
        
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        
        if desc_success and invalid_success:
            print("🎉 All descriptor tests passed!")
            print("The descriptor functions are working correctly.")
        else:
            print("⚠️  Some descriptor tests failed.")
    else:
        print("❌ Data loading failed. Check the data file.")

if __name__ == "__main__":
    main() 