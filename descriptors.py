"""
    descriptors based on stoichiometry and functional groups
    Daniel C. Elton
    License : MIT
"""
from rdkit.Chem import Descriptors
from collections import defaultdict
import numpy as np
from rdkit.Chem.Descriptors import _descList
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit import Chem
from rdkit.Chem import AllChem

# Try importing thermo for heat capacity
try:
    from thermo import Chemical
    THERMO_AVAILABLE = True
except ImportError:
    THERMO_AVAILABLE = False
    print("Warning: thermo not installed. Cv will be 0. Install with: pip install thermo")


def get_num_atom(mol: Chem.Mol, atomic_number: int) -> int:
    """
    Counts the number of atoms of a specified atomic number in a given RDKit molecule.

    Parameters:
        mol (rdkit.Chem.Mol): The molecule object to analyze.
        atomic_number (int): The atomic number of the atom to count.

    Returns:
        int: The number of atoms in the molecule with the specified atomic number.
    """
    num = 0
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if (atom_num == atomic_number):
            num += 1
    return num


def count_substructure(mol: Chem.Mol, substruct_smarts: str) -> int:
    """
    Counts the number of times a given substructure (by SMILES) appears in a molecule.
    
    Parameters:
        mol: RDKit Mol object
        substruct_smarts: SMARTS string of the substructure
    
    Returns:
        int: Number of substructure occurrences in the molecule.
    """
    substruct = Chem.MolFromSmarts(substruct_smarts)
    if substruct is None or mol is None:
        return 0
    return len(mol.GetSubstructMatches(substruct))


def estimate_zpe(mol: Chem.Mol) -> float:
    """
    Fast estimation of Zero-Point Energy (ZPE) based on bond counting.
    Uses approximate vibrational frequencies for common bond types.
    
    Returns:
        float: Estimated ZPE in Hartree
    """
    # Approximate frequencies in cm^-1
    # Very rough averages
    freq_map = {
        'C-H': 2900,
        'N-H': 3300,
        'O-H': 3600,
        'C-C': 1000,
        'C=C': 1600,
        'C#C': 2100,
        'C-N': 1100,
        'C=N': 1600,
        'C#N': 2200,
        'C-O': 1100,
        'C=O': 1700,
        'N-N': 1000,
        'N=N': 1500,
        'N-O': 1000,
        'N=O': 1500,
        'O-O': 900
    }
    
    total_freq_sum = 0.0
    
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        sym1 = a1.GetSymbol()
        sym2 = a2.GetSymbol()
        bond_type = bond.GetBondType()
        
        # Sort symbols alphabetically
        key_atoms = "-".join(sorted([sym1, sym2]))
        
        # Determine bond order string
        if bond_type == Chem.BondType.SINGLE:
            key_bond = "-"
        elif bond_type == Chem.BondType.DOUBLE:
            key_bond = "="
        elif bond_type == Chem.BondType.TRIPLE:
            key_bond = "#"
        elif bond_type == Chem.BondType.AROMATIC:
            key_bond = "-" # Treat aromatic as single for rough frequency or average
            # Ideally averaging between single and double, say 1400
            if key_atoms == "C-C":
                total_freq_sum += 1450
                continue
        else:
            key_bond = "-"

        key = f"{sym1}{key_bond}{sym2}" if sym1 < sym2 else f"{sym2}{key_bond}{sym1}"
        
        # Try to find frequency, default to 1000 if unknown
        freq = freq_map.get(key, 1000)
        total_freq_sum += freq

    # Conversion: E = 0.5 * h * c * nu
    # 1 cm^-1 = 4.55633e-6 Hartree
    # ZPE = 0.5 * sum(nu) * 4.55633e-6
    zpe_hartree = 0.5 * total_freq_sum * 4.55633e-6
    return zpe_hartree


def get_cv(smiles: str) -> float:
    """
    Calculate Volumetric Heat Capacity (Cv) using thermo package.
    Cv = Cp - R (approximation for ideal gas).
    
    Returns:
        float: Cv in J/(mol*K). Returns 0 if thermo not installed.
    """
    if not THERMO_AVAILABLE:
        return 0.0
    try:
        chem = Chemical(smiles)
        # Cp in J/mol/K at 298.15K
        Cp = chem.Cpm
        if Cp is None:
            return 0.0
        # Cv approx Cp - R for ideal gas
        R = 8.314
        return Cp - R
    except:
        return 0.0


def get_homo_lumo(mol: Chem.Mol) -> float:
    """
    Estimate electronic "gap" using Gasteiger Partial Charges.
    Returns the difference between Max Positive and Max Negative charge (Charge Separation).
    This serves as a fast, robust proxy for electronic hardness/polarizability 
    when QM tools (like PySCF/xTB) are unavailable on Windows.
    
    Returns:
        float: Charge gap (proxy for HOMO-LUMO).
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(a.GetProp('_GasteigerCharge')) for a in mol.GetAtoms() if a.HasProp('_GasteigerCharge')]
        if not charges:
            return 0.0
        # Gap ~ Max Positive - Min Negative (most positive - most negative)
        return max(charges) - min(charges)
    except:
        return 0.0


def create_descriptor(smiles: str) -> list:
    """
    Generate a set of molecular descriptors from a SMILES string.

    Parameters:
        smiles (str): The SMILES representation of the molecule.

    Returns:
        list: List of descriptor values
    """
    descriptor = []
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    # Add molecular weight as a descriptor (redundant with later, but per prompt)
    try:
        descriptor.append(Descriptors.MolWt(mol))  # molecular weight
        
        # New Descriptors
        descriptor.append(estimate_zpe(mol)) # ZPE (Hartree)
        descriptor.append(get_cv(smiles))    # Cv (J/mol/K)
        # Note: HOMO-LUMO calculation is slow (seconds/mol). 
        # Comment out the next line if speed is critical for large datasets.
        descriptor.append(get_homo_lumo(mol)) # HOMO-LUMO Gap (Hartree)
    
        substructure_smarts_list = [
            # Single atoms
            '[C]',  # Carbon
            '[H]',  # Hydrogen
            '[N]',  # Nitrogen
            '[O]',  # Oxygen

            # 2-atom fragments (alphabetical by first then second atom)
            '[C]~[C]',
            '[C]~[H]',
            '[C]~[N]',
            '[C]~[O]',
            '[H]~[H]',
            '[H]~[N]',
            '[H]~[O]',
            '[N]~[N]',
            '[N]~[O]',
            '[O]~[O]',

            # 3-atom linear fragments (grouped by central/first atom; within that, second, then third atom alphabetical)
            # C as first atom
            '[C]~[C]~[C]',
            '[C]~[C]~[H]',
            '[C]~[C]~[N]',
            '[C]~[C]~[O]',
            '[C]~[N]~[C]',
            '[C]~[N]~[H]',
            '[C]~[N]~[N]',
            '[C]~[N]~[O]',
            '[C]~[O]~[C]',
            '[C]~[O]~[H]',
            '[C]~[O]~[N]',
            '[C]~[O]~[O]',
            # H as first atom
            '[H]~[C]~[H]',
            '[H]~[C]~[N]',
            '[H]~[C]~[O]',
            '[H]~[N]~[H]',
            '[H]~[N]~[N]',
            '[H]~[N]~[O]',
            '[H]~[O]~[H]',
            '[H]~[O]~[N]',
            '[H]~[O]~[O]',
            # N as first atom
            '[N]~[C]~[N]',
            '[N]~[C]~[O]',
            '[N]~[N]~[N]',
            '[N]~[O]~[N]',
            '[N]~[O]~[O]',
            # O as first atom
            '[O]~[C]~[O]',
            '[O]~[N]~[N]',
            '[O]~[N]~[O]',
            '[O]~[O]~[O]',

            # Three Attachments (groups by central atom with sections for C and N)
            # C Central, sorted lex by number and type of neighbors
            "[C](~[C])(~[N])(~[N])",
            "[C](~[C])(~[N])(~[O])",
            "[C](~[C])(~[O])(~[O])",
            "[C](~[H])(~[H])(~[H])",
            "[C](~[H])(~[H])(~[N])",
            "[C](~[H])(~[H])(~[O])",
            "[C](~[H])(~[N])(~[N])",
            "[C](~[H])(~[N])(~[O])",
            "[C](~[H])(~[O])(~[O])",
            "[C](~[N])(~[N])(~[N])",
            "[C](~[N])(~[N])(~[O])",
            "[C](~[N])(~[O])(~[O])",
            "[C](~[O])(~[O])(~[O])",

            # N Central, sorted
            "[N](~[C])(~[C])(~[C])",
            "[N](~[C])(~[C])(~[H])",
            "[N](~[C])(~[C])(~[N])",
            "[N](~[C])(~[C])(~[O])",
            "[N](~[C])(~[H])(~[H])",
            "[N](~[C])(~[H])(~[N])",
            "[N](~[C])(~[H])(~[O])",
            "[N](~[C])(~[N])(~[N])",
            "[N](~[C])(~[N])(~[O])",
            "[N](~[C])(~[O])(~[O])",
            "[N](~[H])(~[H])(~[H])",
            "[N](~[H])(~[H])(~[N])",
            "[N](~[H])(~[H])(~[O])",
            "[N](~[H])(~[N])(~[N])",
            "[N](~[H])(~[N])(~[O])",
            "[N](~[H])(~[O])(~[O])",
            "[N](~[N])(~[N])(~[N])",
            "[N](~[N])(~[N])(~[O])",
            "[N](~[N])(~[O])(~[O])",
            "[N](~[O])(~[O])(~[O])",

            # Four Attachments (C Central, grouped by composition, increasing number of H, N, O, etc.)
            "[C](~[C])(~[C])(~[C])(~[C])",
            "[C](~[C])(~[C])(~[C])(~[H])",
            "[C](~[C])(~[C])(~[C])(~[N])",
            "[C](~[C])(~[C])(~[C])(~[O])",
            "[C](~[C])(~[C])(~[H])(~[H])",
            "[C](~[C])(~[C])(~[H])(~[N])",
            "[C](~[C])(~[C])(~[H])(~[O])",
            "[C](~[C])(~[C])(~[N])(~[N])",
            "[C](~[C])(~[C])(~[N])(~[O])",
            "[C](~[C])(~[C])(~[O])(~[O])",
            "[C](~[C])(~[H])(~[H])(~[H])",
            "[C](~[C])(~[H])(~[H])(~[N])",
            "[C](~[C])(~[H])(~[H])(~[O])",
            "[C](~[C])(~[H])(~[N])(~[N])",
            "[C](~[C])(~[H])(~[N])(~[O])",
            "[C](~[C])(~[H])(~[O])(~[O])",
            "[C](~[C])(~[N])(~[N])(~[N])",
            "[C](~[C])(~[N])(~[N])(~[O])",
            "[C](~[C])(~[N])(~[O])(~[O])",
            "[C](~[C])(~[O])(~[O])(~[O])",
            "[C](~[H])(~[H])(~[H])(~[H])",
            "[C](~[H])(~[H])(~[H])(~[N])",
            "[C](~[H])(~[H])(~[H])(~[O])",
            "[C](~[H])(~[H])(~[N])(~[N])",
            "[C](~[H])(~[H])(~[N])(~[O])",
            "[C](~[H])(~[H])(~[O])(~[O])",
            "[C](~[H])(~[N])(~[N])(~[N])",
            "[C](~[H])(~[N])(~[N])(~[O])",
            "[C](~[H])(~[N])(~[O])(~[O])",
            "[C](~[H])(~[O])(~[O])(~[O])",
            "[C](~[N])(~[N])(~[N])(~[N])",
            "[C](~[N])(~[N])(~[N])(~[O])",
            "[C](~[N])(~[N])(~[O])(~[O])",
            "[C](~[N])(~[O])(~[O])(~[O])",
            "[C](~[O])(~[O])(~[O])(~[O])"
        ]
        for smarts in substructure_smarts_list:
            descriptor.append(count_substructure(mol, smarts))
        descriptor.append(Descriptors.NumAromaticRings(mol))  # number of aromatic rings
        descriptor.append(Descriptors.NumHAcceptors(mol))  # number of hydrogen bond acceptors
        descriptor.append(Descriptors.NumHDonors(mol))     # number of hydrogen bond donors
    except:
        print(smiles)
    return descriptor
