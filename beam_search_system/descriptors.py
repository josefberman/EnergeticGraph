"""
Molecular descriptor generation module.
Recreated from scratch for the beam search system.
"""

from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, List


def get_num_atom(mol: Chem.Mol, atomic_number: int) -> int:
    """Count atoms of specified atomic number."""
    num = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == atomic_number:
            num += 1
    return num


def count_substructure(mol: Chem.Mol, substruct_smarts: str) -> int:
    """Count occurrences of a substructure pattern."""
    substruct = Chem.MolFromSmarts(substruct_smarts)
    if substruct is None or mol is None:
        return 0
    return len(mol.GetSubstructMatches(substruct))


def estimate_zpe(mol: Chem.Mol) -> float:
    """
    Estimate Zero-Point Energy based on bond counting.
    Returns: ZPE in Hartree
    """
    freq_map = {
        'C-H': 2900, 'N-H': 3300, 'O-H': 3600,
        'C-C': 1000, 'C=C': 1600, 'C#C': 2100,
        'C-N': 1100, 'C=N': 1600, 'C#N': 2200,
        'C-O': 1100, 'C=O': 1700,
        'N-N': 1000, 'N=N': 1500,
        'N-O': 1000, 'N=O': 1500,
        'O-O': 900
    }
    
    total_freq_sum = 0.0
    
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        sym1, sym2 = a1.GetSymbol(), a2.GetSymbol()
        bond_type = bond.GetBondType()
        
        if bond_type == Chem.BondType.SINGLE:
            key_bond = "-"
        elif bond_type == Chem.BondType.DOUBLE:
            key_bond = "="
        elif bond_type == Chem.BondType.TRIPLE:
            key_bond = "#"
        elif bond_type == Chem.BondType.AROMATIC:
            key_bond = "-"
            if sorted([sym1, sym2]) == ['C', 'C']:
                total_freq_sum += 1450
                continue
        else:
            key_bond = "-"
        
        key = f"{sym1}{key_bond}{sym2}" if sym1 < sym2 else f"{sym2}{key_bond}{sym1}"
        freq = freq_map.get(key, 1000)
        total_freq_sum += freq
    
    # Convert to Hartree: 1 cm^-1 = 4.55633e-6 Hartree
    zpe_hartree = 0.5 * total_freq_sum * 4.55633e-6
    return zpe_hartree


def get_cv(smiles: str) -> float:
    """
    Calculate volumetric heat capacity (stub - returns 0 for now).
    Full implementation requires thermo package.
    """
    return 0.0


def get_homo_lumo(mol: Chem.Mol) -> float:
    """
    Estimate electronic gap using Gasteiger charges.
    Returns: Charge separation as proxy for HOMO-LUMO gap
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(a.GetProp('_GasteigerCharge')) 
                   for a in mol.GetAtoms() 
                   if a.HasProp('_GasteigerCharge')]
        if not charges:
            return 0.0
        return max(charges) - min(charges)
    except:
        return 0.0


def create_descriptor(smiles: str) -> Optional[List[float]]:
    """
    Generate molecular descriptor vector from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        List of descriptor values, or None if invalid SMILES
    """
    descriptor = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
    except:
        return None
    
    try:
        # Basic descriptors
        descriptor.append(Descriptors.MolWt(mol))
        descriptor.append(estimate_zpe(mol))
        descriptor.append(get_cv(smiles))
        descriptor.append(get_homo_lumo(mol))
        
        # Substructure patterns (SMARTS)
        substructure_smarts_list = [
            # Single atoms
            '[C]', '[H]', '[N]', '[O]',
            
            # 2-atom fragments
            '[C]~[C]', '[C]~[H]', '[C]~[N]', '[C]~[O]',
            '[H]~[H]', '[H]~[N]', '[H]~[O]',
            '[N]~[N]', '[N]~[O]', '[O]~[O]',
            
            # 3-atom linear fragments
            '[C]~[C]~[C]', '[C]~[C]~[H]', '[C]~[C]~[N]', '[C]~[C]~[O]',
            '[C]~[N]~[C]', '[C]~[N]~[H]', '[C]~[N]~[N]', '[C]~[N]~[O]',
            '[C]~[O]~[C]', '[C]~[O]~[H]', '[C]~[O]~[N]', '[C]~[O]~[O]',
            '[H]~[C]~[H]', '[H]~[C]~[N]', '[H]~[C]~[O]',
            '[H]~[N]~[H]', '[H]~[N]~[N]', '[H]~[N]~[O]',
            '[H]~[O]~[H]', '[H]~[O]~[N]', '[H]~[O]~[O]',
            '[N]~[C]~[N]', '[N]~[C]~[O]', '[N]~[N]~[N]',
            '[N]~[O]~[N]', '[N]~[O]~[O]',
            '[O]~[C]~[O]', '[O]~[N]~[N]', '[O]~[N]~[O]', '[O]~[O]~[O]',
            
            # Three attachments (C and N central)
            "[C](~[C])(~[N])(~[N])", "[C](~[C])(~[N])(~[O])", "[C](~[C])(~[O])(~[O])",
            "[C](~[H])(~[H])(~[H])", "[C](~[H])(~[H])(~[N])", "[C](~[H])(~[H])(~[O])",
            "[C](~[H])(~[N])(~[N])", "[C](~[H])(~[N])(~[O])", "[C](~[H])(~[O])(~[O])",
            "[C](~[N])(~[N])(~[N])", "[C](~[N])(~[N])(~[O])", "[C](~[N])(~[O])(~[O])",
            "[C](~[O])(~[O])(~[O])",
            "[N](~[C])(~[C])(~[C])", "[N](~[C])(~[C])(~[H])", "[N](~[C])(~[C])(~[N])",
            "[N](~[C])(~[C])(~[O])", "[N](~[C])(~[H])(~[H])", "[N](~[C])(~[H])(~[N])",
            "[N](~[C])(~[H])(~[O])", "[N](~[C])(~[N])(~[N])", "[N](~[C])(~[N])(~[O])",
            "[N](~[C])(~[O])(~[O])", "[N](~[H])(~[H])(~[H])", "[N](~[H])(~[H])(~[N])",
            "[N](~[H])(~[H])(~[O])", "[N](~[H])(~[N])(~[N])", "[N](~[H])(~[N])(~[O])",
            "[N](~[H])(~[O])(~[O])", "[N](~[N])(~[N])(~[N])", "[N](~[N])(~[N])(~[O])",
            "[N](~[N])(~[O])(~[O])", "[N](~[O])(~[O])(~[O])",
            
            # Four attachments (C central)
            "[C](~[C])(~[C])(~[C])(~[C])", "[C](~[C])(~[C])(~[C])(~[H])",
            "[C](~[C])(~[C])(~[C])(~[N])", "[C](~[C])(~[C])(~[C])(~[O])",
            "[C](~[C])(~[C])(~[H])(~[H])", "[C](~[C])(~[C])(~[H])(~[N])",
            "[C](~[C])(~[C])(~[H])(~[O])", "[C](~[C])(~[C])(~[N])(~[N])",
            "[C](~[C])(~[C])(~[N])(~[O])", "[C](~[C])(~[C])(~[O])(~[O])",
            "[C](~[C])(~[H])(~[H])(~[H])", "[C](~[C])(~[H])(~[H])(~[N])",
            "[C](~[C])(~[H])(~[H])(~[O])", "[C](~[C])(~[H])(~[N])(~[N])",
            "[C](~[C])(~[H])(~[N])(~[O])", "[C](~[C])(~[H])(~[O])(~[O])",
            "[C](~[C])(~[N])(~[N])(~[N])", "[C](~[C])(~[N])(~[N])(~[O])",
            "[C](~[C])(~[N])(~[O])(~[O])", "[C](~[C])(~[O])(~[O])(~[O])",
            "[C](~[H])(~[H])(~[H])(~[H])", "[C](~[H])(~[H])(~[H])(~[N])",
            "[C](~[H])(~[H])(~[H])(~[O])", "[C](~[H])(~[H])(~[N])(~[N])",
            "[C](~[H])(~[H])(~[N])(~[O])", "[C](~[H])(~[H])(~[O])(~[O])",
            "[C](~[H])(~[N])(~[N])(~[N])", "[C](~[H])(~[N])(~[N])(~[O])",
            "[C](~[H])(~[N])(~[O])(~[O])", "[C](~[H])(~[O])(~[O])(~[O])",
            "[C](~[N])(~[N])(~[N])(~[N])", "[C](~[N])(~[N])(~[N])(~[O])",
            "[C](~[N])(~[N])(~[O])(~[O])", "[C](~[N])(~[O])(~[O])(~[O])",
            "[C](~[O])(~[O])(~[O])(~[O])"
        ]
        
        for smarts in substructure_smarts_list:
            descriptor.append(count_substructure(mol, smarts))
        
        # Additional RDKit descriptors
        descriptor.append(Descriptors.NumAromaticRings(mol))
        descriptor.append(Descriptors.NumHAcceptors(mol))
        descriptor.append(Descriptors.NumHDonors(mol))
        
    except Exception as e:
        print(f"Error creating descriptor for {smiles}: {e}")
        return None
    
    return descriptor
