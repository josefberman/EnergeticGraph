#!/usr/bin/env python3
"""
Script to check the data structure and identify issues
"""

import pandas as pd

def check_data_structure():
    """Check the data structure"""
    print("Checking data structure...")
    
    try:
        # Load data
        df = pd.read_excel("clean_data_imputed.xlsx")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check SMILES column
        if 'SMILES' in df.columns:
            print(f"\nSMILES column info:")
            print(f"  Total SMILES: {len(df['SMILES'])}")
            print(f"  Non-null SMILES: {df['SMILES'].notna().sum()}")
            print(f"  Sample SMILES: {df['SMILES'].iloc[0]}")
        
        # Check property columns
        property_columns = ['Density', 'Detonation velocity', 'Explosion capacity', 
                           'Explosion pressure', 'Explosion heat']
        
        print(f"\nProperty columns:")
        for col in property_columns:
            if col in df.columns:
                print(f"  {col}: {df[col].notna().sum()} non-null values")
                if df[col].notna().sum() > 0:
                    print(f"    Range: {df[col].min():.2f} to {df[col].max():.2f}")
            else:
                print(f"  {col}: NOT FOUND")
        
        # Check for any problematic data
        print(f"\nData types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return False

def test_descriptor_creation():
    """Test descriptor creation with sample data"""
    print("\nTesting descriptor creation...")
    
    try:
        from prediction import create_descriptor
        
        # Load data
        df = pd.read_excel("clean_data_imputed.xlsx")
        
        # Test first few SMILES
        for i in range(min(5, len(df))):
            smiles = df['SMILES'].iloc[i]
            if pd.notna(smiles):
                print(f"  Testing SMILES {i}: {smiles}")
                try:
                    descriptor = create_descriptor(smiles)
                    print(f"    Descriptor: {descriptor}")
                except Exception as e:
                    print(f"    Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in descriptor test: {e}")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("DATA STRUCTURE CHECK")
    print("=" * 50)
    
    # Check data structure
    structure_ok = check_data_structure()
    
    if structure_ok:
        # Test descriptor creation
        descriptor_ok = test_descriptor_creation()
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        
        if descriptor_ok:
            print("✅ Data structure looks good!")
            print("The issue might be in the retraining script.")
        else:
            print("⚠️  Descriptor creation has issues.")
    else:
        print("❌ Data structure has problems.")

if __name__ == "__main__":
    main() 