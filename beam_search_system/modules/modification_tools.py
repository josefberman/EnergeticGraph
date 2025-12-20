"""
Chemical modification tools using RDKit.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional
import logging
import random

logger = logging.getLogger(__name__)


def addition_modification(smiles: str, functional_groups: List[str] = None) -> List[str]:
    """
    Add functional groups to molecule at available positions.
    
    Args:
        smiles: Input SMILES
        functional_groups: List of SMARTS functional groups to add
        
    Returns:
        List of modified SMILES
    """
    if functional_groups is None:
        # Default energetic functional groups
        functional_groups = [
            '[N+](=O)[O-]',  # Nitro group
            'N',  # Amino group
            'N=[N+]=[N-]',  # Azido group
            'C#N',  # Cyano group
        ]
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    results = []
    
    # Find attachment points (C or N atoms with available valence)
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Skip if not C or N
        if atom.GetSymbol() not in ['C', 'N']:
            continue
        
        # Check if atom has hydrogen we can replace
        if atom.GetTotalNumHs() == 0:
            continue
        
        # Try adding each functional group
        for fg_smiles in functional_groups:
            try:
                # Create editable molecule
                edit_mol = Chem.RWMol(mol)
                
                # Add functional group
                fg_mol = Chem.MolFromSmiles(fg_smiles)
                if fg_mol is None:
                    continue
                
                # Simple approach: replace H with functional group
                # This is a simplified version - real implementation would use SMARTS reactions
                combined = Chem.CombineMols(edit_mol, fg_mol)
                
                # Try to get SMILES
                new_smiles = Chem.MolToSmiles(combined)
                if new_smiles and new_smiles != smiles:
                    results.append(new_smiles)
            
            except Exception as e:
                logger.debug(f"Failed to add {fg_smiles} to {smiles}: {e}")
                continue
    
    return list(set(results))[:5]  # Return unique results, max 5


def subtraction_modification(smiles: str) -> List[str]:
    """
    Remove terminal atoms or small groups from molecule.
    
    Args:
        smiles: Input SMILES
        
    Returns:
        List of modified SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    results = []
    
    # Find terminal atoms (degree 1)
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        
        if atom.GetDegree() == 1:
            try:
                edit_mol = Chem.RWMol(mol)
                edit_mol.RemoveAtom(atom_idx)
                
                # Sanitize and get SMILES
                Chem.SanitizeMol(edit_mol)
                new_smiles = Chem.MolToSmiles(edit_mol)
                
                if new_smiles and new_smiles != smiles and edit_mol.GetNumAtoms() > 3:
                    results.append(new_smiles)
            
            except Exception as e:
                logger.debug(f"Failed to remove atom {atom_idx} from {smiles}: {e}")
                continue
    
    return list(set(results))[:3]


def substitution_modification(smiles: str) -> List[str]:
    """
    Substitute atoms or groups in the molecule.
    
    Args:
        smiles: Input SMILES
        
    Returns:
        List of modified SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    results = []
    
    # Define substitution reactions using SMARTS
    reactions = [
        # Replace C-H with C-NO2
        ('[C:1][H]', '[C:1][N+](=O)[O-]'),
        # Replace N-H with N-NO2
        ('[N:1][H]', '[N:1][N+](=O)[O-]'),
        # Replace C-H with C-N3
        ('[C:1][H]', '[C:1]N=[N+]=[N-]'),
    ]
    
    for reactant_smarts, product_smarts in reactions:
        try:
            rxn = AllChem.ReactionFromSmarts(f'{reactant_smarts}>>{product_smarts}')
            products = rxn.RunReactants((mol,))
            
            for product_tuple in products:
                for product in product_tuple:
                    try:
                        Chem.SanitizeMol(product)
                        new_smiles = Chem.MolToSmiles(product)
                        if new_smiles and new_smiles != smiles:
                            results.append(new_smiles)
                    except:
                        continue
        
        except Exception as e:
            logger.debug(f"Substitution reaction failed: {e}")
            continue
    
    return list(set(results))[:5]


def ring_modification(smiles: str) -> List[str]:
    """
    Close chains into rings (cyclization) or open rings into chains.
    
    Args:
        smiles: Input SMILES
        
    Returns:
        List of modified SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    results = []
    
    # 1. Ring Opening: Break rings into chains
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        # Try to open rings
        for ring in ring_info.BondRings():
            if len(ring) < 3:  # Skip very small rings
                continue
            
            # Try opening at different bonds in the ring
            for bond_idx in list(ring)[:2]:  # Try first 2 bonds to avoid too many variants
                try:
                    edit_mol = Chem.RWMol(mol)
                    bond = edit_mol.GetBondWithIdx(bond_idx)
                    
                    # Get atoms
                    atom1_idx = bond.GetBeginAtomIdx()
                    atom2_idx = bond.GetEndAtomIdx()
                    
                    # Remove the bond
                    edit_mol.RemoveBond(atom1_idx, atom2_idx)
                    
                    # Add hydrogens to maintain valency
                    atom1 = edit_mol.GetAtomWithIdx(atom1_idx)
                    atom2 = edit_mol.GetAtomWithIdx(atom2_idx)
                    atom1.SetNumExplicitHs(atom1.GetNumExplicitHs() + 1)
                    atom2.SetNumExplicitHs(atom2.GetNumExplicitHs() + 1)
                    
                    # Try to sanitize and get SMILES
                    Chem.SanitizeMol(edit_mol)
                    new_smiles = Chem.MolToSmiles(edit_mol)
                    
                    if new_smiles and new_smiles != smiles:
                        results.append(new_smiles)
                
                except Exception as e:
                    logger.debug(f"Failed to open ring at bond {bond_idx}: {e}")
                    continue
    
    # 2. Ring Closing: Close chains into rings (cyclization)
    # Find atoms that could form a ring (atoms with hydrogens that are 3-6 bonds apart)
    for atom1_idx in range(mol.GetNumAtoms()):
        atom1 = mol.GetAtomWithIdx(atom1_idx)
        
        # Skip if no hydrogen available
        if atom1.GetTotalNumHs() == 0:
            continue
        
        # Find potential cyclization partners
        for atom2_idx in range(atom1_idx + 1, mol.GetNumAtoms()):
            atom2 = mol.GetAtomWithIdx(atom2_idx)
            
            # Skip if no hydrogen available
            if atom2.GetTotalNumHs() == 0:
                continue
            
            # Check path length (for reasonable ring sizes: 3-7)
            try:
                path_length = len(Chem.GetShortestPath(mol, atom1_idx, atom2_idx))
                
                # Ideal for 5, 6, or 7 membered rings
                if path_length < 3 or path_length > 7:
                    continue
                
                # Try to form a bond
                edit_mol = Chem.RWMol(mol)
                
                # Add bond between atoms
                edit_mol.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
                
                # Remove hydrogens
                a1 = edit_mol.GetAtomWithIdx(atom1_idx)
                a2 = edit_mol.GetAtomWithIdx(atom2_idx)
                if a1.GetNumExplicitHs() > 0:
                    a1.SetNumExplicitHs(a1.GetNumExplicitHs() - 1)
                if a2.GetNumExplicitHs() > 0:
                    a2.SetNumExplicitHs(a2.GetNumExplicitHs() - 1)
                
                # Try to sanitize and get SMILES
                Chem.SanitizeMol(edit_mol)
                new_smiles = Chem.MolToSmiles(edit_mol)
                
                if new_smiles and new_smiles != smiles:
                    results.append(new_smiles)
                    
                    # Limit cyclization attempts to avoid too many variants
                    if len(results) >= 5:
                        break
            
            except Exception as e:
                logger.debug(f"Failed to cyclize atoms {atom1_idx}-{atom2_idx}: {e}")
                continue
        
        if len(results) >= 5:
            break
    
    return list(set(results))[:5]


def apply_all_modifications(smiles: str) -> List[str]:
    """
    Apply all modification strategies to a molecule.
    
    Args:
        smiles: Input SMILES
        
    Returns:
        List of all modified SMILES from all strategies
    """
    all_modifications = []
    
    all_modifications.extend(addition_modification(smiles))
    all_modifications.extend(subtraction_modification(smiles))
    all_modifications.extend(substitution_modification(smiles))
    all_modifications.extend(ring_modification(smiles))
    
    # Remove duplicates and original
    unique_mods = list(set(all_modifications))
    unique_mods = [s for s in unique_mods if s != smiles]
    
    return unique_mods
