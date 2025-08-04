# Explosives Quantum Chemical Database

A comprehensive database of 500 explosives with quantum chemical (DFT) properties calculated using **Psi4** quantum chemistry package.

## Overview

This project generates a comprehensive database of molecular properties for explosives using real quantum chemical calculations. The database includes 500 explosives ordered from most common to less common, with properties calculated using Density Functional Theory (DFT) via the Psi4 quantum chemistry package.

## Properties Calculated

### General Properties
- **Density** - Molecular density calculated from quantum mechanical volume
- **Gas Phase Heat of Formation** - Calculated from atomization energies using DFT
- **Bond Dissociation Energies** - For identifying weak bonds using quantum calculations

### Stability and Sensitivity Properties
- **BDE of Trigger Bonds** - For impact sensitivity assessment
- **Cohesive Energy/Lattice Energy** - Calculated using quantum mechanical approach
- **Equation of State/Compressibility** - Pressure-volume relationships
- **Elastic Moduli** - For mechanical sensitivity (Young's, Shear, Poisson's ratio)
- **Phonon Dispersion** - For thermal stability and decomposition prediction

### Detonation Properties
- **Detonation Heat (Q)** - Calculated from molecular energy
- **Detonation Velocity (D)** - Estimated from quantum mechanical properties
- **Detonation Pressure (P)** - Calculated from density and velocity
- **Oxygen Balance (OB%)** - Percentage oxygen balance

## File Structure

```
QM_database/
├── requirements.txt              # Python dependencies
├── explosives_list.py           # List of 500 explosives with SMILES
├── quantum_calculator.py        # Core quantum calculations using Psi4
├── advanced_calculations.py     # Advanced properties (phonon, EOS, etc.)
├── main_database_generator.py   # Main orchestration script
├── test_psi4.py                # Test script for Psi4 functionality
├── install.py                   # Automated installation script
├── README.md                    # This file
└── results/                     # Output directory (created automatically)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Sufficient RAM (4GB+ recommended for quantum calculations)

### Automated Installation
Run the automated installation script:
```bash
python install.py
```

This will:
1. Check Python version
2. Install all required packages
3. Test imports
4. Create necessary directories
5. Run a quick test calculation

### Manual Installation
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Test the installation:
```bash
python test_psi4.py
```

## Usage

### Quick Test
Test the quantum chemical calculations:
```bash
python test_psi4.py
```

### Generate Database
Run the main database generator:
```bash
python main_database_generator.py
```

This will:
1. Start with a test run of 5 compounds
2. Ask if you want to run the full database (500 compounds)
3. Save results in multiple formats (CSV, Excel)
4. Generate analysis reports

### Custom Calculations
```python
from quantum_calculator import ExplosiveQuantumCalculator
from advanced_calculations import AdvancedExplosiveCalculator

# Initialize calculators
quantum_calc = ExplosiveQuantumCalculator(basis='def2-TZVP', functional='B3LYP')
advanced_calc = AdvancedExplosiveCalculator(basis='def2-TZVP', functional='B3LYP')

# Calculate properties for a single compound
results = quantum_calc.calculate_molecular_properties("CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]", "TNT")
```

## Explosive Categories

The database includes explosives from various categories:
- **Nitroaromatic** - TNT, DNT, etc.
- **Nitramine** - RDX, HMX, etc.
- **Nitrate Ester** - PETN, NG, etc.
- **Peroxide** - TATP, HMTD, etc.
- **Azide** - Lead azide, etc.
- **And many more...**

## Calculation Methods

### Quantum Chemical Framework
- **Package**: Psi4 (Python-based quantum chemistry)
- **Method**: Density Functional Theory (DFT)
- **Functional**: B3LYP (default)
- **Basis Set**: def2-TZVP (default)
- **Geometry Optimization**: Automatic optimization

### Molecular Handling
- **SMILES Parsing**: RDKit for molecular structure
- **3D Coordinate Generation**: RDKit with MMFF optimization
- **Geometry Optimization**: Psi4 automatic optimization

### Property Calculations
- **Density**: Quantum mechanical volume calculation
- **Heat of Formation**: Atomization energy from DFT
- **BDE**: Bond dissociation energies from quantum calculations
- **Detonation Properties**: Empirical relationships from quantum data
- **Elastic Properties**: Quantum mechanical force constants
- **Phonon Properties**: Quantum mechanical vibrational analysis

## Output Files

### Main Results
- `explosives_quantum_database_YYYYMMDD_HHMMSS.csv` - Main database
- `explosives_quantum_database_YYYYMMDD_HHMMSS.xlsx` - Excel format

### Analysis Reports
- `summary_report_YYYYMMDD_HHMMSS.txt` - Generation summary
- `correlations_YYYYMMDD_HHMMSS.csv` - Property correlations
- `strong_correlations_YYYYMMDD_HHMMSS.txt` - Strong correlations
- `category_*_stats_YYYYMMDD_HHMMSS.csv` - Category-specific statistics

### Intermediate Results
- `intermediate_results_*_compounds_YYYYMMDD_HHMMSS.csv` - Progress saves

## Performance Notes

### Computational Requirements
- **Memory**: 4GB+ RAM recommended
- **CPU**: Multi-core recommended for parallel calculations
- **Time**: ~1-5 minutes per compound (depending on size and complexity)

### Optimization Tips
- Use smaller basis sets for faster calculations
- Reduce memory usage for large molecules
- Run calculations in batches
- Use intermediate saves for long runs

## Customization

### Changing Calculation Parameters
```python
# Different basis set
quantum_calc = ExplosiveQuantumCalculator(basis='6-31G*', functional='B3LYP')

# Different functional
quantum_calc = ExplosiveQuantumCalculator(basis='def2-TZVP', functional='PBE0')

# More memory
quantum_calc = ExplosiveQuantumCalculator(max_memory=8000)
```

### Adding New Properties
Extend the `ExplosiveQuantumCalculator` class:
```python
def calculate_custom_property(self, mol):
    # Your custom calculation
    return {'custom_property': value}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

2. **Memory Issues**
   - Reduce `max_memory` parameter
   - Use smaller basis sets
   - Close other applications

3. **Calculation Failures**
   - Check SMILES validity
   - Try different initial geometries
   - Increase convergence thresholds

4. **Slow Performance**
   - Use smaller basis sets
   - Reduce memory usage
   - Run on faster hardware

### Getting Help
- Check the test script: `python test_psi4.py`
- Review error messages in console output
- Check intermediate results for clues

## Scientific Background

### Quantum Chemical Methods
- **DFT**: Density Functional Theory for electronic structure
- **B3LYP**: Hybrid functional combining exact and approximate exchange
- **def2-TZVP**: Triple-zeta basis set with polarization functions

### Property Relationships
- **Detonation Properties**: Empirical relationships from quantum data
- **Sensitivity**: Correlated with trigger bond BDE
- **Stability**: Related to cohesive energy and phonon properties

## License

This project is for educational and research purposes. Please ensure compliance with local regulations regarding explosive materials.

## Contributing

Contributions are welcome! Please:
1. Test your changes thoroughly
2. Update documentation
3. Follow the existing code style
4. Add appropriate error handling

## Acknowledgments

- Psi4 development team for the quantum chemistry package
- RDKit developers for molecular handling capabilities
- Scientific community for empirical relationships and data 