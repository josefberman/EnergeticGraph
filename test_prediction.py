#!/usr/bin/env python3
"""
Test script to verify the prediction function works correctly
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

def test_prediction():
    """Test the prediction function"""
    print("Testing prediction function...")
    
    try:
        from prediction import predict_properties
        
        # Test with a simple molecule (TNT)
        test_smiles = "CC1=CC=C(C=C1)[N+](=O)[O-]"
        
        print(f"Testing prediction for: {test_smiles}")
        result = predict_properties.invoke(test_smiles)
        
        print("✅ Prediction successful!")
        print("Predicted properties:")
        for prop, value in result.items():
            print(f"  {prop}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_descriptor_creation():
    """Test descriptor creation"""
    print("Testing descriptor creation...")
    
    try:
        from prediction import create_descriptor
        
        test_smiles = "CC1=CC=C(C=C1)[N+](=O)[O-]"
        descriptor = create_descriptor(test_smiles)
        
        print(f"✅ Descriptor created successfully!")
        print(f"Descriptor shape: {len(descriptor)}")
        print(f"Descriptor type: {type(descriptor)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Descriptor creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("PREDICTION FUNCTION TEST")
    print("=" * 50)
    
    # Test descriptor creation first
    desc_success = test_descriptor_creation()
    
    if desc_success:
        # Test prediction
        pred_success = test_prediction()
        
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        
        if pred_success:
            print("🎉 All prediction tests passed!")
            print("The prediction system is working correctly.")
        else:
            print("⚠️  Prediction test failed, but descriptor creation works.")
    else:
        print("❌ Descriptor creation failed. Check the system setup.")

if __name__ == "__main__":
    main() 