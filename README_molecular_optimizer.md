# Molecular Optimization Agent System

This system provides an agentic molecular optimization framework using LangGraph and beam search to design energetic materials with desired properties.

## Overview

The molecular optimization agent system consists of:

1. **`molecular_optimizer.py`** - Basic beam search optimization system
2. **`molecular_optimizer_agent.py`** - Enhanced agent using LangGraph with all available tools
3. **`molecular_tools.py`** - Additional molecular manipulation and validation tools
4. **`sample_optimization_input.csv`** - Example input file format

## Features

- **RAG-Based Starting Molecule Discovery**: Automatically finds suitable starting molecules using RAG
- **Beam Search Optimization**: Iterative molecular modification using beam search algorithm
- **RAG-Enhanced Modifications**: Uses RAG to find modification strategies during optimization
- **Property Prediction**: Uses trained ML models to predict energetic material properties
- **Molecular Validation**: Comprehensive validation of molecular structures
- **Energetic Group Detection**: Identifies and counts energetic functional groups
- **LangGraph Integration**: Agent-based decision making for molecular modifications
- **Verbose Search Process**: Detailed logging of the optimization process
- **Addition and Removal Modifications**: Can both add and remove chemical groups for optimization

## Prerequisites

1. Ensure you have the trained models from the main system:
   ```bash
   python main.py  # This will train the models if they don't exist
   ```

2. Required dependencies (same as the main system):
   - rdkit
   - langchain
   - langgraph
   - scikit-learn
   - pandas
   - numpy
   - matplotlib

## Input Format

The system expects a CSV file with the following columns:

### Required Columns:
- At least one target property column from:
  - `Density`
  - `Detonation velocity`
  - `Explosion capacity`
  - `Explosion pressure`
  - `Explosion heat`
  - `Solid phase formation enthalpy`

**Note**: No starting molecule is required. The system will automatically find a suitable starting molecule using RAG based on the target properties.

### Optional Columns:
- `{property}_weight`: Weight for each target property (default: 1.0)

### Example CSV:
```csv
Density,Detonation velocity,Explosion capacity,Explosion pressure,Explosion heat,Solid phase formation enthalpy,Density_weight,Detonation velocity_weight,Explosion capacity_weight,Explosion pressure_weight,Explosion heat_weight,Solid phase formation enthalpy_weight
1.8,8000,0.9,250,1200,60,1.0,1.0,1.0,1.0,1.0,1.0
```

## Usage

### Basic Optimization (molecular_optimizer.py)

```bash
python molecular_optimizer.py
```

This will prompt you for a CSV file path and run the basic beam search optimization.

### Enhanced Agent Optimization (molecular_optimizer_agent.py)

```bash
python molecular_optimizer_agent.py
```

This runs the enhanced version with LangGraph integration and more sophisticated molecular modifications.

## How It Works

### 1. Input Processing
- Reads the CSV file to extract target properties
- Uses RAG to find a suitable starting molecule based on target properties
- Validates the starting molecule structure

### 2. Beam Search Optimization
- **Initialization**: Starts with the RAG-discovered molecule in the beam
- **Iteration**: For each iteration:
  - Uses RAG to find modification strategies based on current properties vs targets
  - Generates modifications using both RAG suggestions and agent-based generation
  - Predicts properties for each modification
  - Calculates fitness scores based on target properties
  - Selects top candidates for the next beam
- **Convergence**: Stops when improvement falls below threshold or max iterations reached

### 3. Molecular Modifications
The system can perform various modifications:

**Addition Strategies:**
- **Functional Group Addition**: Add nitro, azido, nitramine, etc.
- **Ring Modifications**: Create energetic heterocycles
- **Substituent Addition**: Add various chemical groups

**Removal Strategies:**
- **Functional Group Removal**: Remove nitro, azido, hydroxyl, amino, etc.
- **Terminal Atom Removal**: Remove terminal atoms that don't contribute to properties
- **Substituent Simplification**: Remove complex groups to improve performance
- **Stability Improvement**: Remove groups that may cause instability

**Validation**: Each modification is validated for chemical reasonableness

### 4. Property Prediction
Uses trained ML models to predict:
- Density
- Detonation velocity
- Explosion capacity
- Explosion pressure
- Explosion heat
- Solid phase formation enthalpy

### 5. Fitness Scoring
Calculates how close predicted properties are to target values:
```
Score = Σ(weight_i × (1 - |predicted_i - target_i| / target_i))
```

## Output

The system provides:

1. **Console Output**: Verbose search process with:
   - Iteration-by-iteration progress
   - Best scores and molecules
   - Modification descriptions
   - Property predictions

2. **JSON Results File**: Detailed results including:
   - Best molecule found
   - Final properties
   - Search history
   - Optimization statistics

## Configuration

You can modify the optimization parameters in the agent initialization:

```python
agent = MolecularOptimizationAgent(
    beam_width=5,              # Number of candidates to keep in beam
    max_iterations=8,          # Maximum optimization iterations
    convergence_threshold=0.01 # Stop when improvement < threshold
)
```

## Available Tools

The system includes several molecular tools:

- `validate_molecule_structure()`: Validates molecular structures
- `generate_molecular_modifications()`: Generates valid modifications
- `calculate_molecular_descriptors()`: Calculates molecular descriptors
- `check_energetic_functional_groups()`: Identifies energetic groups
- `predict_properties()`: Predicts energetic material properties
- `convert_name_to_smiles()`: Converts chemical names to SMILES

## Example Usage

1. **Create a CSV file** with your desired target properties (no starting molecule needed)
2. **Run the optimization**:
   ```bash
   python molecular_optimizer_agent.py
   ```
3. **Enter the CSV file path** when prompted
4. **Monitor the RAG search and optimization process** in the console
5. **Check the results** in the generated JSON file

## Troubleshooting

### Common Issues:

1. **"Trained models not found"**
   - Run `python main.py` first to train the models

2. **"Could not convert molecule name"**
   - Try using SMILES notation instead of chemical names
   - Check that the chemical name is correct

3. **"No valid modifications found"**
   - The starting molecule may be too complex or unstable
   - Try a simpler starting molecule

4. **Poor optimization results**
   - Adjust the target properties to more realistic values
   - Modify the weights to prioritize certain properties
   - Increase beam width or max iterations

## Advanced Features

### Custom Molecular Modifications
You can extend the `MolecularGenerator` class to add custom modification strategies for both addition and removal operations.

### Property Weighting
Adjust the weights in the CSV file to prioritize certain properties over others.

### Validation Rules
Modify the `MolecularValidator` class to add custom validation rules for your specific use case.

## Safety Notes

- This system is for research purposes only
- Always validate results with experimental data
- Consider safety implications of energetic materials
- Follow proper laboratory safety protocols

## Contributing

To extend the system:
1. Add new molecular modification strategies
2. Implement additional validation rules
3. Create new property prediction models
4. Enhance the fitness scoring function 