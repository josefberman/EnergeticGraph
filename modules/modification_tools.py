"""
Chemical modification tools using RDKit.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Optional
import logging
import random

logger = logging.getLogger(__name__)


def addition_modification(smiles: str, functional_groups: List[str] = None, max_results: int = None) -> List[str]:
    """
    Add functional groups to molecule at available positions.
    Uses proper bond creation to attach groups.
    
    Args:
        smiles: Input SMILES
        functional_groups: List of SMARTS functional groups to add
        max_results: Optional max results (None = no limit)
        
    Returns:
        List of modified SMILES with functional groups properly bonded
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    results = []
    
    # SMILES patterns that can be directly substituted for hydrogen
    # Format: (smarts_pattern, attachment_smiles)
    substituent_patterns = [
        ('[N+](=O)[O-]', '[N+](=O)[O-]'),  # Nitro group -NO2
        ('N', 'N'),  # Amino -NH2
        ('[N-]=[N+]=[N-]', '[N-]=[N+]=[N-]'),  # Azido -N3
        ('C#N', 'C#N'),  # Cyano -CN
        ('O', 'O'),  # Hydroxyl -OH
        ('F', 'F'),  # Fluoro -F
        ('Cl', 'Cl'),  # Chloro -Cl
        ('C(=O)O', 'C(=O)O'),  # Carboxylic acid -COOH
        ('C(=O)N', 'C(=O)N'),  # Amide -CONH2
        ('N=O', 'N=O'),  # Nitroso -NO
    ]
    
    # Find carbon atoms that have hydrogens we can replace
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        
        # Only modify C atoms (safer chemistry)
        if atom.GetSymbol() != 'C':
            continue
        
        # Check if atom has hydrogens to replace
        num_hs = atom.GetTotalNumHs()
        if num_hs == 0:
            continue
        
        # Try each substituent by direct SMILES manipulation
        for pattern_name, substituent in substituent_patterns:
            try:
                # Create a modified SMILES by replacing one H with the substituent
                # Use RWMol for proper bond creation
                edit_mol = Chem.RWMol(mol)
                
                # Add substituent molecule
                sub_mol = Chem.MolFromSmiles(substituent)
                if sub_mol is None:
                    continue
                
                # Find attachment point on substituent (first atom)
                sub_attach_idx = 0
                
                # Add all atoms from substituent to edit_mol
                atom_map = {}
                for sub_atom in sub_mol.GetAtoms():
                    new_idx = edit_mol.AddAtom(sub_atom)
                    atom_map[sub_atom.GetIdx()] = new_idx
                
                # Add all bonds from substituent
                for bond in sub_mol.GetBonds():
                    begin_idx = atom_map[bond.GetBeginAtomIdx()]
                    end_idx = atom_map[bond.GetEndAtomIdx()]
                    edit_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
                
                # Connect substituent to the target atom with a single bond
                edit_mol.AddBond(atom_idx, atom_map[sub_attach_idx], Chem.BondType.SINGLE)
                
                # Sanitize and get SMILES
                try:
                    Chem.SanitizeMol(edit_mol)
                    new_smiles = Chem.MolToSmiles(edit_mol)
                    if new_smiles and new_smiles != smiles:
                        results.append(new_smiles)
                except:
                    pass  # Sanitization failed, skip this modification
                    
            except Exception as e:
                logger.debug(f"Failed to add {pattern_name} to position {atom_idx}: {e}")
                continue
    
    unique_results = list(set(results))
    if max_results:
        return unique_results[:max_results]
    return unique_results


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
    
    return list(set(results))  # No limit - return all valid results


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
    
    return list(set(results))  # No limit - return all valid results


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
        
        if len(results) >= 20:  # Increased limit for more diversity
            break
    
    return list(set(results))  # No limit - return all valid results


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


def generate_diverse_modifications(smiles: str, target_count: int = 20) -> List[str]:
    """
    Generate a diverse set of molecular modifications to reach target_count.
    
    Uses multiple strategies:
    1. Apply all basic modification types
    2. If not enough, apply depth-2 modifications (modify the modifications)
    3. Shuffle and return diverse set
    
    Args:
        smiles: Input SMILES string
        target_count: Target number of modifications to generate
        
    Returns:
        List of modified SMILES (up to target_count, deduplicated)
    """
    import random
    
    all_mods = set()
    seen_parents = {smiles}
    
    # Level 1: Direct modifications
    level1 = apply_all_modifications(smiles)
    all_mods.update(level1)
    logger.info(f"Level 1 modifications: {len(level1)} unique candidates")
    
    # If we have enough, return early
    if len(all_mods) >= target_count:
        result = list(all_mods)
        random.shuffle(result)
        return result[:target_count]
    
    # Level 2: Modifications of modifications (depth-2)
    # Sample from level 1 to avoid combinatorial explosion
    level1_sample = random.sample(level1, min(len(level1), 5)) if level1 else []
    
    for parent in level1_sample:
        if parent in seen_parents:
            continue
        seen_parents.add(parent)
        
        try:
            level2 = apply_all_modifications(parent)
            # Filter out already seen
            new_mods = [m for m in level2 if m not in all_mods and m != smiles]
            all_mods.update(new_mods)
            
            if len(all_mods) >= target_count:
                break
        except Exception as e:
            logger.debug(f"Failed level 2 modification from {parent}: {e}")
            continue
    
    logger.info(f"Total modifications after level 2: {len(all_mods)} candidates")
    
    # Shuffle for diversity
    result = list(all_mods)
    random.shuffle(result)
    return result[:target_count]
