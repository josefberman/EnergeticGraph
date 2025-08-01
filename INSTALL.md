# Installation Guide

## Quick Installation

### Option 1: Automated Installation (Recommended)
```bash
python install.py
```

### Option 2: Manual Installation (Step by Step)
```bash
# Step 1: Install RDKit (choose one method)
pip install rdkit
# OR
conda install -c conda-forge rdkit
# OR
pip install rdkit-pypi

# Step 2: Install remaining dependencies
pip install -r requirements_no_rdkit.txt
```

### Option 3: All-in-One Manual Installation
```bash
pip install -r requirements.txt
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- OpenAI API key (for the enhanced agent)

## RDKit Warning Fix

The RDKit warning you encountered is harmless but can be suppressed. The installation script automatically adds warning suppression to all Python files. If you see this warning:

```
RuntimeWarning: to-Python converter for boost::shared_ptr<RDKit::FilterHierarchyMatcher> already registered
```

**Solutions:**

1. **Automatic (Recommended)**: The installation script adds warning suppression to all files
2. **Manual**: Add this to the top of your Python scripts:
   ```python
   import warnings
   warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")
   ```
3. **Environment Variable**: Add to your `.env` file:
   ```
   PYTHONWARNINGS=ignore::RuntimeWarning:importlib._bootstrap:*
   ```

## Setup Steps

1. **Install Dependencies**:
   ```bash
   python install.py
   ```

2. **Configure Environment**:
   - Edit `.env` file
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

3. **Ensure Trained Models**:
   - Run `python main.py` to train models (if not already done)
   - Or ensure `trained_models/` directory exists with `.pkl` files

4. **Run the System**:
   ```bash
   # Basic optimizer (requires starting molecule)
   python molecular_optimizer.py
   
   # Enhanced agent (no starting molecule needed)
   python molecular_optimizer_agent.py
   ```

## Troubleshooting

### Common Issues

1. **RDKit Installation Fails**:
   ```bash
   # Method 1: Try different package names
   pip install rdkit
   # OR
   pip install rdkit-pypi
   
   # Method 2: Use conda (recommended for RDKit)
   conda install -c conda-forge rdkit
   
   # Method 3: Platform-specific installation
   # Windows: conda install -c conda-forge rdkit
   # macOS: brew install rdkit && pip install rdkit
   # Linux: sudo apt-get install python3-rdkit
   ```

2. **OpenAI API Errors**:
   - Check your API key in `.env` file
   - Ensure you have sufficient credits

3. **Missing Trained Models**:
   - Run `python main.py` to train models
   - Check that `trained_models/` directory exists

4. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version`

### Platform-Specific Notes

**Windows**:
- Use `python install.py` (not `python3`)
- If RDKit fails, try: `conda install -c conda-forge rdkit`

**macOS**:
- Use `python3 install.py`
- If RDKit fails, try: `brew install rdkit`

**Linux**:
- Use `python3 install.py`
- May need: `sudo apt-get install python3-dev`

## Verification

After installation, test the system:

```bash
# Test basic functionality
python -c "from rdkit import Chem; print('RDKit working')"

# Test molecular tools
python -c "from molecular_tools import validate_molecule_structure; print('Molecular tools working')"

# Test prediction
python -c "from prediction import predict_properties; print('Prediction working')"
```

## Next Steps

1. Read `README_molecular_optimizer.md` for usage instructions
2. Create a CSV file with your target properties
3. Run the optimization system
4. Check the results in the generated JSON files

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your OpenAI API key is valid
4. Check that trained models exist in `trained_models/` 