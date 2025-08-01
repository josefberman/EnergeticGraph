# Suppress RDKit warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

from typing import List, Dict, Any, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from langchain_core.tools import tool
import numpy as np

class MolecularValidator:
    """Validates molecular structures for energetic materials"""
    
    def __init__(self):
        # Initialize RDKit filter catalog for drug-likeness
        try:
            self.filter_catalog = FilterCatalog()
            # Try to add catalogs if the method exists
            if hasattr(self.filter_catalog, 'AddCatalog'):
                self.filter_catalog.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
                self.filter_catalog.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            else:
                # For newer RDKit versions, use different approach
                self.filter_catalog = None
        except Exception as e:
            # If filter catalog fails, continue without it
            self.filter_catalog = None
    
    def validate_molecule(self, smiles: str) -> Dict[str, Any]:
        """Comprehensive molecular validation"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "error": "Invalid SMILES"}
        
        results = {
            "valid": True,
            "warnings": [],
            "properties": {}
        }
        
        # Basic molecular properties
        results["properties"]["molecular_weight"] = Descriptors.ExactMolWt(mol)
        results["properties"]["logp"] = Descriptors.MolLogP(mol)
        results["properties"]["hbd"] = Descriptors.NumHDonors(mol)
        results["properties"]["hba"] = Descriptors.NumHAcceptors(mol)
        results["properties"]["rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)
        results["properties"]["aromatic_rings"] = Descriptors.NumAromaticRings(mol)
        results["properties"]["heteroatoms"] = Descriptors.NumHeteroatoms(mol)
        
        # Energetic material specific checks
        results["properties"]["nitrogen_count"] = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 7])
        results["properties"]["oxygen_count"] = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 8])
        results["properties"]["carbon_count"] = len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 6])
        
        # Check for energetic functional groups
        energetic_groups = self._count_energetic_groups(mol)
        results["properties"]["energetic_groups"] = energetic_groups
        
        # Validation checks
        if results["properties"]["molecular_weight"] > 1000:
            results["warnings"].append("Molecular weight > 1000 Da")
        
        if results["properties"]["molecular_weight"] < 50:
            results["warnings"].append("Molecular weight < 50 Da")
        
        if results["properties"]["nitrogen_count"] == 0:
            results["warnings"].append("No nitrogen atoms (important for energetic materials)")
        
        if results["properties"]["nitrogen_count"] > 20:
            results["warnings"].append("High nitrogen content (>20 atoms)")
        
        # Check for problematic substructures
        problematic_patterns = self._check_problematic_patterns(mol)
        if problematic_patterns:
            results["warnings"].extend(problematic_patterns)
        
        # Check filter catalog
        if self.filter_catalog is not None:
            try:
                filter_matches = self.filter_catalog.GetMatches(mol)
                if filter_matches:
                    results["warnings"].append(f"Matches {len(filter_matches)} problematic substructure filters")
            except Exception as e:
                # Skip filter catalog check if it fails
                pass
        
        return results
    
    def _count_energetic_groups(self, mol: Chem.Mol) -> Dict[str, int]:
        """Count energetic functional groups"""
        groups = {
            "nitro": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7+](=[#8])[#8-]'))),
            "azido": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7-]=[#7+]=[#7]'))),
            "nitroso": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]=[#8]'))),
            "nitramine": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7][#7+](=[#8])[#8-]'))),
            "tetrazole": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1nnn[nH]1'))),
            "triazole": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1nncn1'))),
            "furazan": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1n[nH]oc1'))),
            "oxadiazole": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1n[nH]oc1'))),
        }
        return groups
    
    def _check_problematic_patterns(self, mol: Chem.Mol) -> List[str]:
        """Check for problematic molecular patterns"""
        warnings = []
        
        # Check for unstable patterns
        unstable_patterns = [
            ("peroxide", '[#8][#8]', "Peroxide linkage detected"),
            ("epoxide", 'C1OC1', "Epoxide ring detected"),
            ("azide_chain", '[#7-][#7+]#[#7]', "Azide group detected"),
            ("fulminate", '[#6-]#[#7+][#8]', "Fulminate group detected"),
        ]
        
        for name, pattern, warning in unstable_patterns:
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            if matches:
                warnings.append(warning)
        
        return warnings

class MolecularGenerator:
    """Generates molecular modifications for optimization"""
    
    def __init__(self):
        self.common_substituents = {
            'nitro': '[N+](=O)[O-]',
            'amino': 'N',
            'hydroxyl': 'O',
            'fluoro': 'F',
            'chloro': 'Cl',
            'methyl': 'C',
            'ethyl': 'CC',
            'azido': '[N-]=[N+]=N',
            'nitroso': 'N=O',
            'cyano': 'C#N',
            'carboxyl': 'C(=O)O',
            'ester': 'C(=O)OC',
            'ether': 'OC',
            'thiol': 'S',
            'sulfone': 'S(=O)(=O)',
            'nitramine': 'N[N+](=O)[O-]',
            'tetrazole': 'c1nnn[nH]1',
            'triazole': 'c1nncn1',
            'imidazole': 'c1ncnc1',
            'pyrazole': 'c1n[nH]cc1',
            'furazan': 'c1n[nH]oc1',
            'oxadiazole': 'c1n[nH]oc1',
            'dinitro': '[N+](=O)[O-][N+](=O)[O-]',
            'trinitro': '[N+](=O)[O-][N+](=O)[O-][N+](=O)[O-]',
        }
    
    def generate_modifications(self, smiles: str, max_modifications: int = 10) -> List[Dict[str, Any]]:
        """Generate valid molecular modifications"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        modifications = []
        validator = MolecularValidator()
        
        # Strategy 1: Add energetic groups to available positions
        for group_name, group_smiles in self.common_substituents.items():
            if len(modifications) >= max_modifications:
                break
                
            try:
                # Try to add the group to different positions
                modified_smiles = self._add_substituent(smiles, group_smiles)
                if modified_smiles and modified_smiles != smiles:
                    # Validate the modification
                    validation = validator.validate_molecule(modified_smiles)
                    if validation["valid"]:
                        modifications.append({
                            "smiles": modified_smiles,
                            "description": f"Added {group_name} group",
                            "group": group_name,
                            "validation": validation
                        })
            except:
                continue
        
        # Strategy 2: Remove existing substituents
        removal_modifications = self._generate_removal_modifications(smiles)
        modifications.extend(removal_modifications[:max_modifications - len(modifications)])
        
        # Strategy 3: Ring modifications
        ring_modifications = self._generate_ring_modifications(smiles)
        modifications.extend(ring_modifications[:max_modifications - len(modifications)])
        
        return modifications
    
    def _add_substituent(self, smiles: str, substituent_smiles: str) -> Optional[str]:
        """Add a substituent to a molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            substituent_mol = Chem.MolFromSmiles(substituent_smiles)
            
            if mol is None or substituent_mol is None:
                return None
            
            # Find a suitable position (carbon with hydrogen)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    # Check for hydrogen neighbors
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add substituent
                            for i, substituent_atom in enumerate(substituent_mol.GetAtoms()):
                                new_atom_idx = modified_mol.AddAtom(substituent_atom)
                                if i == 0:  # First atom connects to carbon
                                    modified_mol.AddBond(atom.GetIdx(), new_atom_idx, Chem.BondType.SINGLE)
                            
                            # Add bonds within substituent
                            for bond in substituent_mol.GetBonds():
                                begin_idx = modified_mol.GetNumAtoms() - substituent_mol.GetNumAtoms() + bond.GetBeginAtomIdx()
                                end_idx = modified_mol.GetNumAtoms() - substituent_mol.GetNumAtoms() + bond.GetEndAtomIdx()
                                modified_mol.AddBond(begin_idx, end_idx, bond.GetBondType())
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def _remove_substituent(self, smiles: str, substituent_pattern: str) -> Optional[str]:
        """Remove a substituent from a molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Create pattern molecule for matching
            pattern_mol = Chem.MolFromSmarts(substituent_pattern)
            if pattern_mol is None:
                return None
            
            # Find matches of the substituent pattern
            matches = mol.GetSubstructMatches(pattern_mol)
            
            for match in matches:
                try:
                    # Create modified molecule
                    modified_mol = Chem.RWMol(mol)
                    
                    # Find the attachment point (atom connected to the rest of the molecule)
                    attachment_atom = None
                    substituent_atoms = set(match)
                    
                    for atom_idx in match:
                        atom = mol.GetAtomWithIdx(atom_idx)
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetIdx() not in substituent_atoms:
                                attachment_atom = neighbor.GetIdx()
                                break
                        if attachment_atom is not None:
                            break
                    
                    if attachment_atom is None:
                        continue
                    
                    # Remove substituent atoms (in reverse order to avoid index issues)
                    atoms_to_remove = sorted(match, reverse=True)
                    for atom_idx in atoms_to_remove:
                        modified_mol.RemoveAtom(atom_idx)
                    
                    # Add hydrogen to the attachment point if needed
                    attachment_atom_new = modified_mol.GetAtomWithIdx(attachment_atom)
                    if attachment_atom_new.GetNumImplicitHs() == 0 and attachment_atom_new.GetNumExplicitHs() == 0:
                        # Add explicit hydrogen
                        h_atom = modified_mol.AddAtom(Chem.Atom('H'))
                        modified_mol.AddBond(attachment_atom, h_atom, Chem.BondType.SINGLE)
                    
                    Chem.SanitizeMol(modified_mol)
                    modified_smiles = Chem.MolToSmiles(modified_mol)
                    
                    if modified_smiles and modified_smiles != smiles:
                        return modified_smiles
                        
                except:
                    continue
            
            return None
        except:
            return None
    
    def _generate_ring_modifications(self, smiles: str) -> List[Dict[str, Any]]:
        """Generate ring-based modifications"""
        modifications = []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return modifications
            
            # Find aromatic rings
            aromatic_rings = mol.GetSubstructMatches(Chem.MolFromSmarts('a'))
            
            for ring_match in aromatic_rings:
                # Try to add nitrogen atoms to create energetic heterocycles
                for atom_idx in ring_match:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() == 'C':
                        # Try replacing carbon with nitrogen
                        modified_mol = Chem.RWMol(mol)
                        modified_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(7)
                        
                        try:
                            Chem.SanitizeMol(modified_mol)
                            modified_smiles = Chem.MolToSmiles(modified_mol)
                            if modified_smiles != smiles:
                                validator = MolecularValidator()
                                validation = validator.validate_molecule(modified_smiles)
                                if validation["valid"]:
                                    modifications.append({
                                        "smiles": modified_smiles,
                                        "description": "Replaced carbon with nitrogen in aromatic ring",
                                        "group": "nitrogen_substitution",
                                        "validation": validation
                                    })
                        except:
                            continue
            
            return modifications
        except:
            return modifications
    
    def _generate_removal_modifications(self, smiles: str) -> List[Dict[str, Any]]:
        """Generate modifications by removing existing substituents"""
        modifications = []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return modifications
            
            validator = MolecularValidator()
            
            # Define patterns for removable substituents
            removable_patterns = {
                'nitro_group': '[#6][#7+](=[#8])[#8-]',
                'azido_group': '[#7-]=[#7+]=[#7]',
                'nitroso_group': '[#7]=[#8]',
                'nitramine_group': '[#7][#7+](=[#8])[#8-]',
                'cyano_group': '[#6]#[#7]',
                'hydroxyl_group': '[#6][#8][#1]',
                'amino_group': '[#6][#7]([#1])[#1]',
                'fluoro_group': '[#6][#9]',
                'chloro_group': '[#6][#17]',
                'methyl_group': '[#6][#6]([#1])([#1])[#1]',
                'ethyl_group': '[#6][#6][#6]([#1])([#1])[#1]',
                'carboxyl_group': '[#6](=[#8])[#8][#1]',
                'ester_group': '[#6](=[#8])[#8][#6]',
                'ether_group': '[#6][#8][#6]',
                'thiol_group': '[#6][#16][#1]',
                'sulfone_group': '[#6][#16](=[#8])(=[#8])[#6]',
                'peroxide_group': '[#8][#8]',
                'fulminate_group': '[#6-]#[#7+][#8]',
            }
            
            for group_name, pattern in removable_patterns.items():
                try:
                    # Try to remove the substituent
                    modified_smiles = self._remove_substituent(smiles, pattern)
                    if modified_smiles and modified_smiles != smiles:
                        # Validate the modification
                        validation = validator.validate_molecule(modified_smiles)
                        if validation["valid"]:
                            modifications.append({
                                "smiles": modified_smiles,
                                "description": f"Removed {group_name}",
                                "group": f"removed_{group_name}",
                                "validation": validation
                            })
                except:
                    continue
            
            # Also try removing individual atoms that might be problematic
            individual_atom_removals = self._generate_individual_atom_removals(smiles)
            modifications.extend(individual_atom_removals)
            
            return modifications
        except:
            return modifications
    
    def _generate_individual_atom_removals(self, smiles: str) -> List[Dict[str, Any]]:
        """Generate modifications by removing individual atoms"""
        modifications = []
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return modifications
            
            validator = MolecularValidator()
            
            # Try removing terminal atoms (atoms with only one neighbor)
            for atom in mol.GetAtoms():
                if atom.GetDegree() == 1:  # Terminal atom
                    try:
                        # Create modified molecule
                        modified_mol = Chem.RWMol(mol)
                        
                        # Find the atom to remove
                        atom_idx = atom.GetIdx()
                        
                        # Get the neighbor (attachment point)
                        neighbor = atom.GetNeighbors()[0]
                        neighbor_idx = neighbor.GetIdx()
                        
                        # Remove the terminal atom
                        modified_mol.RemoveAtom(atom_idx)
                        
                        # Add hydrogen to the neighbor if needed
                        neighbor_new = modified_mol.GetAtomWithIdx(neighbor_idx)
                        if neighbor_new.GetNumImplicitHs() == 0 and neighbor_new.GetNumExplicitHs() == 0:
                            h_atom = modified_mol.AddAtom(Chem.Atom('H'))
                            modified_mol.AddBond(neighbor_idx, h_atom, Chem.BondType.SINGLE)
                        
                        Chem.SanitizeMol(modified_mol)
                        modified_smiles = Chem.MolToSmiles(modified_mol)
                        
                        if modified_smiles and modified_smiles != smiles:
                            # Validate the modification
                            validation = validator.validate_molecule(modified_smiles)
                            if validation["valid"]:
                                atom_symbol = atom.GetSymbol()
                                modifications.append({
                                    "smiles": modified_smiles,
                                    "description": f"Removed terminal {atom_symbol} atom",
                                    "group": f"removed_{atom_symbol}",
                                    "validation": validation
                                })
                    except:
                        continue
            
            return modifications
        except:
            return modifications

@tool
def validate_molecule_structure(smiles: str) -> Dict[str, Any]:
    """
    Validates a molecular structure for energetic materials applications.
    
    Args:
        smiles: SMILES representation of the molecule
        
    Returns:
        Dictionary containing validation results, warnings, and molecular properties
    """
    validator = MolecularValidator()
    return validator.validate_molecule(smiles)

@tool
def generate_molecular_modifications(smiles: str, max_modifications: int = 10) -> List[Dict[str, Any]]:
    """
    Generates valid molecular modifications for energetic materials optimization.
    
    Args:
        smiles: SMILES representation of the starting molecule
        max_modifications: Maximum number of modifications to generate
        
    Returns:
        List of dictionaries containing modified SMILES, descriptions, and validation results
    """
    generator = MolecularGenerator()
    return generator.generate_modifications(smiles, max_modifications)

@tool
def calculate_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """
    Calculates comprehensive molecular descriptors for energetic materials.
    
    Args:
        smiles: SMILES representation of the molecule
        
    Returns:
        Dictionary containing calculated molecular descriptors
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}
    
    descriptors = {
        "molecular_weight": Descriptors.ExactMolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "aromatic_rings": Descriptors.NumAromaticRings(mol),
        "heteroatoms": Descriptors.NumHeteroatoms(mol),
        "nitrogen_count": len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]),
        "oxygen_count": len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]),
        "carbon_count": len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]),
        "hydrogen_count": len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 1]),
        "fluorine_count": len([a for a in mol.GetAtoms() if a.GetAtomicNum() == 9]),
        "tpsa": Descriptors.TPSA(mol),
        "molar_refractivity": Descriptors.MolMR(mol),
        "fraction_csp3": Descriptors.FractionCsp3(mol),
        "heavy_atom_count": Descriptors.HeavyAtomCount(mol),
        "ring_count": Descriptors.RingCount(mol),
        "spiro_atom_count": Descriptors.SpiroAtomCount(mol),
        "bridgehead_atom_count": Descriptors.BridgeheadAtomCount(mol),
    }
    
    # Calculate oxygen balance
    n_O = descriptors["oxygen_count"]
    n_C = descriptors["carbon_count"]
    n_H = descriptors["hydrogen_count"]
    n_atoms = mol.GetNumAtoms()
    
    if n_atoms > 0:
        descriptors["oxygen_balance_100"] = 100 * (n_O - 2 * n_C - n_H / 2) / n_atoms
    else:
        descriptors["oxygen_balance_100"] = 0.0
    
    # Calculate N/C ratio
    if n_C > 0:
        descriptors["nitrogen_carbon_ratio"] = descriptors["nitrogen_count"] / n_C
    else:
        descriptors["nitrogen_carbon_ratio"] = 0.0
    
    return descriptors

@tool
def check_energetic_functional_groups(smiles: str) -> Dict[str, int]:
    """
    Identifies and counts energetic functional groups in a molecule.
    
    Args:
        smiles: SMILES representation of the molecule
        
    Returns:
        Dictionary containing counts of various energetic functional groups
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES"}
    
    groups = {
        "nitro_groups": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6][#7+](=[#8])[#8-]'))),
        "azido_groups": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7-]=[#7+]=[#7]'))),
        "nitroso_groups": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]=[#8]'))),
        "nitramine_groups": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7][#7+](=[#8])[#8-]'))),
        "tetrazole_rings": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1nnn[nH]1'))),
        "triazole_rings": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1nncn1'))),
        "imidazole_rings": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1ncnc1'))),
        "pyrazole_rings": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1n[nH]cc1'))),
        "furazan_rings": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1n[nH]oc1'))),
        "oxadiazole_rings": len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1n[nH]oc1'))),
        "fulminate_groups": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6-]#[#7+][#8]'))),
        "cyano_groups": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#6]#[#7]'))),
        "peroxide_groups": len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#8][#8]'))),
    }
    
    return groups 