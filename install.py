#!/usr/bin/env python3
"""
Molecular Optimization Agent System Installation Script
Handles the RDKit warning and provides a clean installation process.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install requirements with RDKit warning suppression"""
    print("📦 Installing dependencies...")
    
    # Suppress RDKit warnings during installation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            # First try to install RDKit separately
            print("Installing RDKit...")
            rdkit_result = subprocess.run([
                sys.executable, "-m", "pip", "install", "rdkit"
            ], capture_output=True, text=True)
            
            if rdkit_result.returncode != 0:
                print("⚠️  RDKit installation failed, trying alternative...")
                # Try alternative RDKit package
                rdkit_result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "rdkit-pypi"
                ], capture_output=True, text=True)
                
                if rdkit_result.returncode != 0:
                    print("❌ RDKit installation failed. Please install manually:")
                    print("   Option 1: pip install rdkit")
                    print("   Option 2: conda install -c conda-forge rdkit")
                    print("   Option 3: pip install rdkit-pypi")
                    return False
            
            # Install remaining requirements
            print("Installing remaining dependencies...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                return True
            else:
                print("❌ Error installing dependencies:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error during installation: {e}")
            return False

def create_env_file():
    """Create .env file template"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file template...")
        env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize model settings
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0

# Optional: RAG settings
RAG_MAX_DOCS=100
RAG_TOP_K=10

# Optional: Optimization settings
BEAM_WIDTH=5
MAX_ITERATIONS=8
CONVERGENCE_THRESHOLD=0.01
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env file created. Please add your OpenAI API key.")
    else:
        print("✅ .env file already exists")

def check_trained_models():
    """Check if trained models exist"""
    models_dir = Path("trained_models")
    if not models_dir.exists():
        print("⚠️  Trained models directory not found.")
        print("   You'll need to run main.py first to train the models.")
        print("   Or ensure the trained_models/ directory exists with required .pkl files.")
    else:
        print("✅ Trained models directory found")

def create_sample_data():
    """Create sample data directory"""
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    sample_csv = Path("sample_optimization_input.csv")
    if sample_csv.exists():
        import shutil
        shutil.copy(sample_csv, sample_dir / "sample_optimization_input.csv")
        print("✅ Sample data copied to sample_data/")

def suppress_rdkit_warnings():
    """Create a script to suppress RDKit warnings"""
    warning_suppression = """# Add this to the top of your Python scripts to suppress RDKit warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

# Or add this to your .env file:
# PYTHONWARNINGS=ignore::RuntimeWarning:importlib._bootstrap:*
"""
    
    with open("rdkit_warning_suppression.py", "w") as f:
        f.write(warning_suppression)
    print("✅ Created rdkit_warning_suppression.py for warning suppression")

def main():
    """Main installation function"""
    print("=" * 50)
    print("Molecular Optimization Agent System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Check trained models
    check_trained_models()
    
    # Create sample data
    create_sample_data()
    
    # Suppress RDKit warnings
    suppress_rdkit_warnings()
    
    print("\n" + "=" * 50)
    print("Installation Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Add your OpenAI API key to the .env file")
    print("2. Ensure trained models exist in ./trained_models/")
    print("3. Run the system:")
    print("   - Basic: python molecular_optimizer.py")
    print("   - Enhanced: python molecular_optimizer_agent.py")
    print("\nTo suppress RDKit warnings, add this to your scripts:")
    print("import warnings")
    print("warnings.filterwarnings('ignore', category=RuntimeWarning, module='importlib._bootstrap')")
    print("\nFor more information, see README_molecular_optimizer.md")

if __name__ == "__main__":
    main() 