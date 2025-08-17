#!/usr/bin/env python3
"""
Script to properly install PyMuPDF and fix the fitz module issue
"""

import subprocess
import sys

def install_pymupdf():
    """Install PyMuPDF properly"""
    print("Installing PyMuPDF...")
    
    try:
        # Try to install PyMuPDF
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "PyMuPDF>=1.23.0"
        ])
        print("✅ PyMuPDF installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyMuPDF: {e}")
        return False

def test_fitz_import():
    """Test if fitz can be imported properly"""
    print("Testing fitz import...")
    
    try:
        import fitz
        print("✅ fitz module imported successfully")
        
        # Test basic functionality
        print("Testing fitz functionality...")
        doc = fitz.open()
        print("✅ fitz basic functionality working")
        return True
        
    except ImportError as e:
        print(f"❌ fitz import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ fitz functionality test failed: {e}")
        return False

def fix_fitz_issue():
    """Try to fix the fitz module issue"""
    print("Attempting to fix fitz module issue...")
    
    try:
        # Try importing PyMuPDF directly
        import PyMuPDF
        print("✅ PyMuPDF imported successfully")
        
        # Try importing fitz from PyMuPDF
        from PyMuPDF import fitz
        print("✅ fitz imported from PyMuPDF successfully")
        return True
        
    except Exception as e:
        print(f"❌ Could not fix fitz issue: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 50)
    print("PyMuPDF Installation and Fix Script")
    print("=" * 50)
    
    # Install PyMuPDF
    if not install_pymupdf():
        print("Failed to install PyMuPDF")
        return
    
    # Test fitz import
    if not test_fitz_import():
        print("fitz import failed, trying to fix...")
        if not fix_fitz_issue():
            print("Could not fix fitz issue")
            return
    
    print("\n" + "=" * 50)
    print("PyMuPDF Installation Complete!")
    print("=" * 50)
    print("\nYou can now run the molecular optimization system.")
    print("If you still see fitz errors, try:")
    print("1. Restart your Python environment")
    print("2. Run: pip install --upgrade PyMuPDF")
    print("3. Check if there are any conflicting PDF libraries")

if __name__ == "__main__":
    main() 