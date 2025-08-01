#!/usr/bin/env python3
"""
Comprehensive test script to verify all fixes work correctly
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

def test_tool_calls():
    """Test all tool calls to ensure they use .invoke() method"""
    print("Testing tool calls...")
    
    try:
        from prediction import predict_properties, convert_name_to_smiles
        from molecular_tools import validate_molecule_structure, generate_molecular_modifications
        
        # Test predict_properties
        result1 = predict_properties.invoke("CC1=CC=C(C=C1)[N+](=O)[O-]")
        print("✅ predict_properties.invoke() working")
        
        # Test convert_name_to_smiles
        result2 = convert_name_to_smiles.invoke("TNT")
        print("✅ convert_name_to_smiles.invoke() working")
        
        # Test validate_molecule_structure
        result3 = validate_molecule_structure.invoke("CC1=CC=C(C=C1)[N+](=O)[O-]")
        print("✅ validate_molecule_structure.invoke() working")
        
        # Test generate_molecular_modifications
        result4 = generate_molecular_modifications.invoke("CC1=CC=C(C=C1)[N+](=O)[O-]", max_modifications=3)
        print("✅ generate_molecular_modifications.invoke() working")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool call test failed: {e}")
        return False

def test_filter_catalog():
    """Test that FilterCatalog doesn't cause errors"""
    print("Testing FilterCatalog...")
    
    try:
        from molecular_tools import MolecularValidator
        
        validator = MolecularValidator()
        print("✅ MolecularValidator initialized without FilterCatalog errors")
        
        # Test validation
        result = validator.validate_molecule("CC1=CC=C(C=C1)[N+](=O)[O-]")
        print("✅ Molecular validation working")
        
        return True
        
    except Exception as e:
        print(f"❌ FilterCatalog test failed: {e}")
        return False

def test_rag_functionality():
    """Test RAG functionality"""
    print("Testing RAG functionality...")
    
    try:
        from RAG import retrieve_context
        
        result = retrieve_context.invoke("energetic materials")
        print("✅ RAG functionality working")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False

def test_optimization_agent():
    """Test the optimization agent initialization"""
    print("Testing optimization agent...")
    
    try:
        from molecular_optimizer_agent import MolecularOptimizationAgent
        
        agent = MolecularOptimizationAgent(beam_width=3, max_iterations=2)
        print("✅ Optimization agent initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization agent test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("COMPREHENSIVE FIX VERIFICATION TEST")
    print("=" * 60)
    
    tests = [
        ("Tool Calls", test_tool_calls),
        ("FilterCatalog", test_filter_catalog),
        ("RAG Functionality", test_rag_functionality),
        ("Optimization Agent", test_optimization_agent),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The system should now work without errors or warnings.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 