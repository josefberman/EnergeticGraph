"""Utility functions for molecular operations.

This module contains shared utilities for SMILES conversion,
molecular structure manipulation, and ASE atoms handling.
"""
from typing import Tuple, List
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
import re


def smiles_to_mol_3d(smiles: str, n_confs: int = 8) -> Tuple[Chem.Mol, List[int]]:
    """Convert SMILES to 3D molecule with multiple conformers.
    
    Args:
        smiles: SMILES string
        n_confs: Number of conformers to generate
    
    Returns:
        Tuple of (molecule with 3D coords, list of conformer IDs)
    
    Raises:
        ValueError: If SMILES is invalid
    """
    m0 = Chem.MolFromSmiles(smiles)
    if m0 is None:
        raise ValueError("Bad SMILES")
    mol = Chem.AddHs(m0)
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.5
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant="MMFF94s")
    return mol, list(cids)


def molconf_to_ase_atoms(mol: Chem.Mol, cid: int) -> Atoms:
    """Convert RDKit molecule conformer to ASE Atoms object.
    
    Args:
        mol: RDKit molecule with conformers
        cid: Conformer ID to convert
    
    Returns:
        ASE Atoms object with molecular geometry
    """
    conf = mol.GetConformer(cid)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = conf.GetPositions()
    return Atoms(symbols=symbols, positions=coords)


def is_smiles(text: str) -> bool:
    """Check if text looks like SMILES notation.
    
    Args:
        text: Text to check
    
    Returns:
        True if text appears to be SMILES notation
    """
    if not text or len(text) < 2:
        return False
    
    # SMILES typically contains specific characters
    smiles_chars = set('CNOPSFClBrI[]()=#@+-0123456789cnops')
    text_chars = set(text)
    
    # If most characters are SMILES-like, it's probably SMILES
    common_chars = text_chars & smiles_chars
    if len(common_chars) < len(text_chars) * 0.7:
        return False
    
    # Try to parse it
    try:
        mol = Chem.MolFromSmiles(text)
        return mol is not None
    except:
        return False


def extract_smiles_from_text(text: str) -> List[str]:
    """Extract potential SMILES strings from text.
    
    Args:
        text: Text containing potential SMILES
    
    Returns:
        List of extracted SMILES strings
    """
    # Simple pattern matching for SMILES-like strings
    # This is a heuristic - not perfect
    potential_smiles = []
    
    # Split by whitespace and newlines
    tokens = re.split(r'\s+', text)
    
    for token in tokens:
        if len(token) > 3 and is_smiles(token):
            potential_smiles.append(token)
    
    return potential_smiles


def extract_molecule_names(text: str) -> List[str]:
    """Extract potential molecule names from text.
    
    Args:
        text: Text containing potential molecule names
    
    Returns:
        List of extracted molecule names
    """
    # Common energetic material name patterns
    patterns = [
        r'\b([A-Z]{3,})\b',  # Acronyms (TNT, RDX, etc.)
        r'\b(\d,\d[,-]\w+)\b',  # Numbered compounds (1,3,5-trinitrobenzene)
        r'\b([A-Z][a-z]+-\w+)\b',  # Dash-separated (Nitro-compounds)
    ]
    
    names = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        names.extend(matches)
    
    return list(set(names))  # Remove duplicates
