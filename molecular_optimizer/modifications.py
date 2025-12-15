"""Molecular modification strategies for energetic materials.

This module provides molecular transformation methods to generate
candidate molecules for optimization.
"""
from typing import Optional
from rdkit import Chem


class MolecularModifier:
    """Applies molecular modifications to generate candidates."""
    
    def add_nitro_group(self, smiles: str) -> Optional[str]:
        """Add a nitro group (-NO2) to the molecule.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Modified SMILES or None if modification fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a carbon atom with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add nitro group
                            n_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            o1_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            o2_atom = modified_mol.AddAtom(Chem.Atom('O'))
                            
                            # Add bonds
                            modified_mol.AddBond(atom.GetIdx(), n_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n_atom, o1_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n_atom, o2_atom, Chem.BondType.SINGLE)
                            
                            # Set formal charges
                            modified_mol.GetAtomWithIdx(n_atom).SetFormalCharge(1)
                            modified_mol.GetAtomWithIdx(o2_atom).SetFormalCharge(-1)
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def add_azido_group(self, smiles: str) -> Optional[str]:
        """Add an azido group (-N3) to the molecule.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Modified SMILES or None if modification fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a carbon atom with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetSymbol() == 'H':
                            # Create modified molecule
                            modified_mol = Chem.RWMol(mol)
                            
                            # Remove hydrogen
                            modified_mol.RemoveAtom(neighbor.GetIdx())
                            
                            # Add azido group
                            n1_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            n2_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            n3_atom = modified_mol.AddAtom(Chem.Atom('N'))
                            
                            # Add bonds
                            modified_mol.AddBond(atom.GetIdx(), n1_atom, Chem.BondType.SINGLE)
                            modified_mol.AddBond(n1_atom, n2_atom, Chem.BondType.DOUBLE)
                            modified_mol.AddBond(n2_atom, n3_atom, Chem.BondType.DOUBLE)
                            
                            # Set formal charges
                            modified_mol.GetAtomWithIdx(n1_atom).SetFormalCharge(0)
                            modified_mol.GetAtomWithIdx(n2_atom).SetFormalCharge(1)
                            modified_mol.GetAtomWithIdx(n3_atom).SetFormalCharge(-1)
                            
                            Chem.SanitizeMol(modified_mol)
                            return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def add_nitramine_group(self, smiles: str) -> Optional[str]:
        """Add a nitramine group to the molecule.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Modified SMILES or None if modification fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Look for secondary amine (-NH-)
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'N' and atom.GetTotalNumHs() >= 1:
                    modified_mol = Chem.RWMol(mol)
                    
                    # Add NO2 group attached to nitrogen
                    n_atom = modified_mol.AddAtom(Chem.Atom('N'))
                    o1_atom = modified_mol.AddAtom(Chem.Atom('O'))
                    o2_atom = modified_mol.AddAtom(Chem.Atom('O'))
                    
                    # Add bonds
                    modified_mol.AddBond(atom.GetIdx(), n_atom, Chem.BondType.SINGLE)
                    modified_mol.AddBond(n_atom, o1_atom, Chem.BondType.DOUBLE)
                    modified_mol.AddBond(n_atom, o2_atom, Chem.BondType.SINGLE)
                    
                    # Set formal charges
                    modified_mol.GetAtomWithIdx(n_atom).SetFormalCharge(1)
                    modified_mol.GetAtomWithIdx(o2_atom).SetFormalCharge(-1)
                    
                    Chem.SanitizeMol(modified_mol)
                    return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def add_tetrazole_group(self, smiles: str) -> Optional[str]:
        """Add a tetrazole ring to the molecule.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Modified SMILES or None if modification fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Build a simple tetrazole ring (c1nnnn1)
            tetrazole_smiles = 'c1nnnn1'
            tetrazole = Chem.MolFromSmiles(tetrazole_smiles)
            
            if tetrazole is None:
                return None
            
            # Combine molecules (simple approach)
            combined = Chem.CombineMols(mol, tetrazole)
            return Chem.MolToSmiles(combined)
            
        except:
            return None
    
    def remove_nitro_group(self, smiles: str) -> Optional[str]:
        """Remove a nitro group from the molecule.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Modified SMILES or None if modification fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find nitro groups ([N+](=O)[O-])
            pattern = Chem.MolFromSmarts('[N+](=O)[O-]')
            if pattern is None:
                return None
            
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                return None
            
            # Remove first nitro group found
            modified_mol = Chem.RWMol(mol)
            nitro_atoms = matches[0]
            
            # Remove in reverse order to avoid index shifting
            for atom_idx in sorted(nitro_atoms, reverse=True):
                modified_mol.RemoveAtom(atom_idx)
            
            Chem.SanitizeMol(modified_mol)
            return Chem.MolToSmiles(modified_mol)
            
        except:
            return None
    
    def substitute_hydrogen(self, smiles: str) -> Optional[str]:
        """Substitute hydrogen with a more energetic group.
        
        Args:
            smiles: Input SMILES string
        
        Returns:
            Modified SMILES or None if modification fails
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Find a carbon with hydrogen
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 0:
                    modified_mol = Chem.RWMol(mol)
                    
                    # Add fluorine (simple substitution)
                    f_atom = modified_mol.AddAtom(Chem.Atom('F'))
                    modified_mol.AddBond(atom.GetIdx(), f_atom, Chem.BondType.SINGLE)
                    
                    Chem.SanitizeMol(modified_mol)
                    return Chem.MolToSmiles(modified_mol)
            
            return None
        except:
            return None
    
    def apply_modification_from_rag(
        self,
        smiles: str,
        modification_type: str
    ) -> Optional[str]:
        """Apply a modification based on RAG suggestion.
        
        Args:
            smiles: Input SMILES string
            modification_type: Type of modification to apply
        
        Returns:
            Modified SMILES or None if modification fails
        """
        if modification_type == 'nitro_addition':
            return self.add_nitro_group(smiles)
        elif modification_type == 'azido_addition':
            return self.add_azido_group(smiles)
        elif modification_type == 'nitramine_addition':
            return self.add_nitramine_group(smiles)
        elif modification_type == 'tetrazole_addition':
            return self.add_tetrazole_group(smiles)
        elif modification_type == 'nitro_removal':
            return self.remove_nitro_group(smiles)
        elif modification_type == 'hydrogen_substitution':
            return self.substitute_hydrogen(smiles)
        else:
            return None
