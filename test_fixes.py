#!/usr/bin/env python3
"""
Test script to verify fixes for LangChain deprecation warning and PyMuPDF issues
"""

import warnings

# Suppress RDKit warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

def test_rag_functionality():
    """Test RAG functionality with error handling"""
    print("Testing RAG functionality...")
    
    try:
        from RAG import retrieve_context
        
        # Test the tool invocation (should use .invoke() instead of __call__)
        result = retrieve_context.invoke("energetic materials explosives")
        
        if isinstance(result, list):
            print("✅ RAG functionality working")
            print(f"   Found {len(result)} results")
            return True
        else:
            print("❌ RAG returned unexpected result type")
            return False
            
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False

def test_convert_name_to_smiles():
    """Test convert_name_to_smiles tool"""
    print("Testing convert_name_to_smiles tool...")
    
    try:
        from prediction import convert_name_to_smiles
        
        # Test the tool invocation (should use .invoke() instead of __call__)
        result = convert_name_to_smiles.invoke("TNT")
        
        if isinstance(result, str) and result != "Did not convert":
            print("✅ convert_name_to_smiles tool working")
            print(f"   Result: {result}")
            return True
        else:
            print("❌ convert_name_to_smiles returned unexpected result")
            return False
            
    except Exception as e:
        print(f"❌ convert_name_to_smiles test failed: {e}")
        return False

def test_pymupdf_installation():
    """Test PyMuPDF installation"""
    print("Testing PyMuPDF installation...")
    
    try:
        import fitz
        print("✅ PyMuPDF installed successfully")
        return True
    except ImportError as e:
        print(f"❌ PyMuPDF not installed: {e}")
        print("   Installing PyMuPDF...")
        
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "PyMuPDF>=1.23.0"])
            import fitz
            print("✅ PyMuPDF installed successfully")
            return True
        except Exception as install_error:
            print(f"❌ Failed to install PyMuPDF: {install_error}")
            return False

def test_langchain_tools():
    """Test LangChain tool functionality"""
    print("Testing LangChain tools...")
    
    try:
        from langchain_core.tools import tool
        
        @tool
        def test_tool(query: str) -> str:
            return f"Test result for: {query}"
        
        # Test the new .invoke() method
        result = test_tool.invoke("test query")
        print("✅ LangChain tool .invoke() method working")
        return True
        
    except Exception as e:
        print(f"❌ LangChain tool test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Fixes for LangChain and PyMuPDF Issues")
    print("=" * 50)
    
    # Test PyMuPDF
    pymupdf_success = test_pymupdf_installation()
    
    # Test LangChain tools
    langchain_success = test_langchain_tools()
    
    # Test RAG functionality
    rag_success = test_rag_functionality()
    
    # Test convert_name_to_smiles tool
    convert_success = test_convert_name_to_smiles()
    
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    
    if pymupdf_success and langchain_success and rag_success and convert_success:
        print("🎉 All fixes working correctly!")
        print("\nThe system should now run without the previous errors.")
    else:
        print("⚠️  Some issues remain:")
        if not pymupdf_success:
            print("   - PyMuPDF installation issue")
        if not langchain_success:
            print("   - LangChain tool issue")
        if not rag_success:
            print("   - RAG functionality issue")
        if not convert_success:
            print("   - convert_name_to_smiles tool issue")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 