#!/usr/bin/env python3
"""
Installation script for Explosives Quantum Chemical Database
Sets up the environment and installs required dependencies
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version}")
        return True

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Core packages
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "joblib>=1.1.0",
        "psutil>=5.8.0",
        "openpyxl>=3.0.9"
    ]
    
    # Scientific packages
    scientific_packages = [
        "ase>=3.22.0",
        "phonopy>=2.20.0",
        "pymatgen>=2022.8.0"
    ]
    
    # PySCF (quantum chemistry)
    pyscf_packages = [
        "pyscf>=2.0.0"
    ]
    
    # RDKit (molecular handling)
    rdkit_packages = [
        "rdkit>=2022.9.1"
    ]
    
    all_packages = core_packages + scientific_packages + pyscf_packages + rdkit_packages
    
    failed_packages = []
    
    for package in all_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Failed to install {len(failed_packages)} packages:")
        for package in failed_packages:
            print(f"  - {package}")
        return False
    else:
        print("\n✅ All packages installed successfully!")
        return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing imports...")
    
    required_modules = [
        "numpy",
        "pandas", 
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "joblib",
        "psutil",
        "openpyxl",
        "ase",
        "phonopy",
        "pymatgen",
        "pyscf",
        "rdkit"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import {len(failed_imports)} modules:")
        for module in failed_imports:
            print(f"  - {module}")
        return False
    else:
        print("\n✅ All modules imported successfully!")
        return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "results",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_config_file():
    """Create a configuration file"""
    print("\n⚙️  Creating configuration file...")
    
    config_content = """# Explosives Quantum Chemical Database Configuration

# Calculation Parameters
BASIS_SET = "def2-TZVP"
FUNCTIONAL = "B3LYP"
MAX_MEMORY = 4000  # MB

# Output Settings
SAVE_INTERMEDIATE_RESULTS = True
INTERMEDIATE_SAVE_INTERVAL = 10  # Save every N compounds

# Performance Settings
USE_PARALLEL = False
NUM_CORES = 1

# File Paths
RESULTS_DIR = "results"
LOGS_DIR = "logs"
TEMP_DIR = "temp"

# Database Settings
MAX_COMPOUNDS = None  # None for all 500 compounds
START_INDEX = 0
"""
    
    with open("config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Configuration file created: config.py")

def run_test():
    """Run a quick test to verify installation"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test basic imports
        from explosives_list import get_explosives_list
        from quantum_calculator import ExplosiveQuantumCalculator
        
        # Get explosives list
        explosives = get_explosives_list()
        print(f"✅ Loaded {len(explosives)} explosives")
        
        # Test calculator initialization
        calc = ExplosiveQuantumCalculator()
        print("✅ Quantum calculator initialized")
        
        # Test with a simple molecule
        test_result = calc.calculate_molecular_properties(
            "C[N+](=O)[O-]",  # Nitromethane
            "Test_Molecule"
        )
        
        if test_result:
            print("✅ Test calculation successful")
            return True
        else:
            print("❌ Test calculation failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 60)
    print("EXPLOSIVES QUANTUM CHEMICAL DATABASE - INSTALLATION")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install packages
    if not install_requirements():
        print("\n❌ Installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check the errors above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create config file
    create_config_file()
    
    # Run test
    if run_test():
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Run: python test_calculations.py")
        print("2. Run: python main_database_generator.py")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print("\n❌ Installation test failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 