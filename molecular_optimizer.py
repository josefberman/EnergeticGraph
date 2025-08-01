# Suppress RDKit warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="importlib._bootstrap")

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from prediction import predict_properties, convert_name_to_smiles
import os
from dotenv import load_dotenv
import json
from dataclasses import dataclass, asdict
from collections import deque
import heapq

load_dotenv()

@dataclass
class MoleculeCandidate:
    """Represents a molecule candidate in the beam search"""
    smiles: str
    name: str
    properties: Dict[str, float]
    score: float
    parent_smiles: Optional[str] = None
    modification_description: str = ""

class MolecularModifier:
    """Handles molecular modifications for energetic materials optimization"""
    
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
            'pyrazole': 'c1n[nH]cc1'
        }
        
        self.energetic_groups = [
            '[N+](=O)[O-]',  # Nitro
            '[N-]=[N+]=N',   # Azido
            'N=O',           # Nitroso
            'N[N+](=O)[O-]', # Nitramine
            'c1nnn[nH]1',    # Tetrazole
            'c1nncn1',       # Triazole
            'c1ncnc1',       # Imidazole
            'c1n[nH]cc1'     # Pyrazole
        ]
    
    def get_valid_modifications(self, smiles: str) -> List[Tuple[str, str, str]]:
        """
        Generate valid molecular modifications for a given SMILES
        Returns: List of (modified_smiles, modification_description, substituent_name)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        modifications = []
        
        # Strategy 1: Add substituents
        modifications.extend(self._get_addition_modifications(smiles))
        
        # Strategy 2: Remove substituents
        modifications.extend(self._get_removal_modifications(smiles))
        
        return modifications
    
    def _get_addition_modifications(self, smiles: str) -> List[Tuple[str, str, str]]:
        """Generate modifications by adding substituents"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        modifications = []
        
        # Get all hydrogen atoms that can be substituted
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'H':
                continue
                
            # Get hydrogen neighbors
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    # Try different substituents
                    for name, substituent_smiles in self.common_substituents.items():
                        try:
                            # Create a modified molecule by replacing H with substituent
                            modified_mol = Chem.RWMol(mol)
                            
                            # Find the hydrogen atom to replace
                            h_atom_idx = None
                            for bond in mol.GetBonds():
                                if (bond.GetBeginAtomIdx() == atom.GetIdx() and 
                                    mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() == 'H'):
                                    h_atom_idx = bond.GetEndAtomIdx()
                                    break
                                elif (bond.GetEndAtomIdx() == atom.GetIdx() and 
                                      mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() == 'H'):
                                    h_atom_idx = bond.GetBeginAtomIdx()
                                    break
                            
                            if h_atom_idx is not None:
                                # Remove the hydrogen atom
                                modified_mol.RemoveAtom(h_atom_idx)
                                
                                # Add the substituent
                                substituent_mol = Chem.MolFromSmiles(substituent_smiles)
                                if substituent_mol is not None:
                                    # Create a bond between the atom and substituent
                                    substituent_atoms = substituent_mol.GetAtoms()
                                    if len(substituent_atoms) > 0:
                                        # Add the substituent atoms
                                        new_atom_idx = modified_mol.AddAtom(substituent_atoms[0])
                                        modified_mol.AddBond(atom.GetIdx(), new_atom_idx, Chem.BondType.SINGLE)
                                        
                                        # Add remaining atoms in substituent
                                        for i in range(1, len(substituent_atoms)):
                                            new_atom_idx = modified_mol.AddAtom(substituent_atoms[i])
                                            # Add bonds based on substituent structure
                                            if i == 1:  # Connect to first atom
                                                modified_mol.AddBond(new_atom_idx - 1, new_atom_idx, Chem.BondType.SINGLE)
                                        
                                        # Sanitize the molecule
                                        Chem.SanitizeMol(modified_mol)
                                        modified_smiles = Chem.MolToSmiles(modified_mol)
                                        
                                        if modified_smiles and modified_smiles != smiles:
                                            modifications.append((
                                                modified_smiles,
                                                f"Added {name} group to {atom.GetSymbol()}",
                                                name
                                            ))
                        except:
                            continue
        
        return modifications
    
    def _get_removal_modifications(self, smiles: str) -> List[Tuple[str, str, str]]:
        """Generate modifications by removing substituents"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        modifications = []
        
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
                modified_smiles = self._remove_substituent(smiles, pattern)
                if modified_smiles and modified_smiles != smiles:
                    modifications.append((
                        modified_smiles,
                        f"Removed {group_name}",
                        f"removed_{group_name}"
                    ))
            except:
                continue
        
        # Also try removing terminal atoms
        modifications.extend(self._get_terminal_atom_removals(smiles))
        
        return modifications
    
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
    
    def _get_terminal_atom_removals(self, smiles: str) -> List[Tuple[str, str, str]]:
        """Generate modifications by removing terminal atoms"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        modifications = []
        
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
                        atom_symbol = atom.GetSymbol()
                        modifications.append((
                            modified_smiles,
                            f"Removed terminal {atom_symbol} atom",
                            f"removed_{atom_symbol}"
                        ))
                except:
                    continue
        
        return modifications
    
    def add_energetic_groups(self, smiles: str) -> List[Tuple[str, str, str]]:
        """Add energetic functional groups to the molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        modifications = []
        
        # Try adding energetic groups to available positions
        for group_name, group_smiles in self.common_substituents.items():
            if group_name in ['nitro', 'azido', 'nitroso', 'nitramine', 'tetrazole', 'triazole']:
                try:
                    # Create a modified molecule
                    modified_mol = Chem.RWMol(mol)
                    
                    # Find a suitable position (carbon atom with hydrogen)
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() == 'C':
                            # Check if it has hydrogen neighbors
                            has_hydrogen = False
                            for neighbor in atom.GetNeighbors():
                                if neighbor.GetSymbol() == 'H':
                                    has_hydrogen = True
                                    break
                            
                            if has_hydrogen:
                                # Add the energetic group
                                group_mol = Chem.MolFromSmiles(group_smiles)
                                if group_mol is not None:
                                    # Simple addition - replace a hydrogen
                                    for neighbor in atom.GetNeighbors():
                                        if neighbor.GetSymbol() == 'H':
                                            # Remove hydrogen
                                            modified_mol.RemoveAtom(neighbor.GetIdx())
                                            
                                            # Add group atoms
                                            for group_atom in group_mol.GetAtoms():
                                                new_atom_idx = modified_mol.AddAtom(group_atom)
                                                if group_atom.GetIdx() == 0:  # First atom
                                                    modified_mol.AddBond(atom.GetIdx(), new_atom_idx, Chem.BondType.SINGLE)
                                                else:
                                                    # Add bonds within the group
                                                    for bond in group_mol.GetBonds():
                                                        if bond.GetBeginAtomIdx() == group_atom.GetIdx():
                                                            modified_mol.AddBond(
                                                                new_atom_idx, 
                                                                new_atom_idx + bond.GetEndAtomIdx() - group_atom.GetIdx(),
                                                                bond.GetBondType()
                                                            )
                                            
                                            try:
                                                Chem.SanitizeMol(modified_mol)
                                                modified_smiles = Chem.MolToSmiles(modified_mol)
                                                if modified_smiles and modified_smiles != smiles:
                                                    modifications.append((
                                                        modified_smiles,
                                                        f"Added {group_name} group",
                                                        group_name
                                                    ))
                                            except:
                                                continue
                                            break
                                break
                except:
                    continue
        
        return modifications

class BeamSearchOptimizer:
    """Implements beam search for molecular optimization"""
    
    def __init__(self, beam_width: int = 5, max_iterations: int = 10):
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.modifier = MolecularModifier()
        self.visited_smiles = set()
    
    def calculate_score(self, properties: Dict[str, float], target_properties: Dict[str, float], 
                       weights: Dict[str, float]) -> float:
        """Calculate fitness score based on how close properties are to targets"""
        score = 0.0
        
        for prop_name, target_value in target_properties.items():
            if prop_name in properties and prop_name in weights:
                current_value = properties[prop_name]
                # Normalize the difference (assuming positive values)
                if target_value > 0:
                    normalized_diff = abs(current_value - target_value) / target_value
                    score += weights[prop_name] * (1.0 - normalized_diff)
        
        return score
    
    def optimize_molecule(self, starting_smiles: str, target_properties: Dict[str, float], 
                         weights: Dict[str, float]) -> List[MoleculeCandidate]:
        """Run beam search optimization"""
        
        # Initialize beam with starting molecule
        starting_properties = predict_properties(starting_smiles)
        starting_score = self.calculate_score(starting_properties, target_properties, weights)
        
        beam = [MoleculeCandidate(
            smiles=starting_smiles,
            name="Starting molecule",
            properties=starting_properties,
            score=starting_score
        )]
        
        self.visited_smiles.add(starting_smiles)
        
        best_candidates = []
        
        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1} ===")
            print(f"Current beam size: {len(beam)}")
            
            # Generate candidates from current beam
            all_candidates = []
            
            for candidate in beam:
                print(f"  Modifying: {candidate.smiles[:50]}... (score: {candidate.score:.4f})")
                
                # Get modifications
                modifications = self.modifier.get_valid_modifications(candidate.smiles)
                modifications.extend(self.modifier.add_energetic_groups(candidate.smiles))
                
                for modified_smiles, description, group_name in modifications:
                    if modified_smiles in self.visited_smiles:
                        continue
                    
                    try:
                        # Predict properties
                        properties = predict_properties(modified_smiles)
                        score = self.calculate_score(properties, target_properties, weights)
                        
                        new_candidate = MoleculeCandidate(
                            smiles=modified_smiles,
                            name=f"{candidate.name} + {group_name}",
                            properties=properties,
                            score=score,
                            parent_smiles=candidate.smiles,
                            modification_description=description
                        )
                        
                        all_candidates.append(new_candidate)
                        self.visited_smiles.add(modified_smiles)
                        
                        print(f"    -> {modified_smiles[:50]}... (score: {score:.4f})")
                        
                    except Exception as e:
                        print(f"    -> Error predicting properties: {e}")
                        continue
            
            # Select top candidates for next beam
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            beam = all_candidates[:self.beam_width]
            
            # Keep track of best candidates
            best_candidates.extend(all_candidates[:self.beam_width])
            best_candidates.sort(key=lambda x: x.score, reverse=True)
            best_candidates = best_candidates[:self.beam_width * 2]
            
            print(f"Best score in this iteration: {beam[0].score if beam else 'N/A'}")
            
            # Early stopping if no improvement
            if not beam:
                print("No valid modifications found. Stopping.")
                break
        
        return best_candidates

class MolecularOptimizationAgent:
    """Main agent class for molecular optimization"""
    
    def __init__(self):
        self.model = ChatOpenAI(model='gpt-4o', temperature=0)
        self.optimizer = BeamSearchOptimizer(beam_width=5, max_iterations=8)
    
    def process_csv_input(self, csv_file_path: str) -> Dict[str, Any]:
        """Process CSV input and run optimization"""
        
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        if len(df) == 0:
            return {"error": "CSV file is empty"}
        
        # Get first row (assuming single molecule optimization)
        row = df.iloc[0]
        
        # Extract molecule information
        molecule_input = row.get('molecule', row.get('smiles', row.get('name', '')))
        if pd.isna(molecule_input):
            return {"error": "No molecule information found in CSV"}
        
        # Extract target properties
        target_properties = {}
        weights = {}
        
        property_columns = ['Density', 'Detonation velocity', 'Explosion capacity', 
                           'Explosion pressure', 'Explosion heat', 'Solid phase formation enthalpy']
        
        for prop in property_columns:
            if prop in df.columns and not pd.isna(row[prop]):
                target_properties[prop] = float(row[prop])
                # Default weight of 1.0, can be customized
                weight_col = f"{prop}_weight"
                weights[prop] = float(row[weight_col]) if weight_col in df.columns else 1.0
        
        if not target_properties:
            return {"error": "No target properties found in CSV"}
        
        # Convert molecule name to SMILES if needed
        if not self._is_smiles(molecule_input):
            smiles = convert_name_to_smiles(molecule_input)
            if smiles == 'Did not convert':
                return {"error": f"Could not convert molecule name '{molecule_input}' to SMILES"}
        else:
            smiles = molecule_input
        
        print(f"Starting optimization for molecule: {smiles}")
        print(f"Target properties: {target_properties}")
        print(f"Weights: {weights}")
        
        # Run optimization
        best_candidates = self.optimizer.optimize_molecule(smiles, target_properties, weights)
        
        # Prepare results
        results = {
            "starting_molecule": smiles,
            "target_properties": target_properties,
            "weights": weights,
            "best_candidates": []
        }
        
        for i, candidate in enumerate(best_candidates[:5]):  # Top 5 results
            results["best_candidates"].append({
                "rank": i + 1,
                "smiles": candidate.smiles,
                "name": candidate.name,
                "properties": candidate.properties,
                "score": candidate.score,
                "parent_smiles": candidate.parent_smiles,
                "modification_description": candidate.modification_description
            })
        
        return results
    
    def _is_smiles(self, text: str) -> bool:
        """Check if text looks like SMILES notation"""
        # Simple heuristic: SMILES typically contains brackets, numbers, and chemical symbols
        smiles_chars = set('[](){}1234567890=#@+-.\\/')
        text_chars = set(text)
        return len(text_chars.intersection(smiles_chars)) > 0 or 'C' in text_chars

def main():
    """Main function to run the molecular optimization agent"""
    
    # Check if trained models exist
    if not os.path.exists('./trained_models/'):
        print("Trained models not found. Please run the main.py first to train the models.")
        return
    
    # Initialize the agent
    agent = MolecularOptimizationAgent()
    
    # Get CSV file path from user
    csv_file_path = input("Enter the path to your CSV file: ").strip()
    
    if not os.path.exists(csv_file_path):
        print(f"Error: File '{csv_file_path}' not found.")
        return
    
    # Process the optimization
    print("\n" + "="*60)
    print("MOLECULAR OPTIMIZATION AGENT")
    print("="*60)
    
    results = agent.process_csv_input(csv_file_path)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nStarting molecule: {results['starting_molecule']}")
    print(f"Target properties: {results['target_properties']}")
    
    print(f"\nTop {len(results['best_candidates'])} optimized molecules:")
    print("-" * 60)
    
    for candidate in results['best_candidates']:
        print(f"\nRank {candidate['rank']}:")
        print(f"  SMILES: {candidate['smiles']}")
        print(f"  Name: {candidate['name']}")
        print(f"  Score: {candidate['score']:.4f}")
        print(f"  Modification: {candidate['modification_description']}")
        print(f"  Properties:")
        for prop_name, value in candidate['properties'].items():
            target = results['target_properties'].get(prop_name, 'N/A')
            print(f"    {prop_name}: {value:.4f} (target: {target})")
    
    # Save results to file
    output_file = "optimization_results.json"
    with open(output_file, 'w') as f:
        # Convert dataclass objects to dictionaries
        json_results = results.copy()
        json_results['best_candidates'] = [
            {k: v for k, v in candidate.items() if k != 'parent_smiles'} 
            for candidate in json_results['best_candidates']
        ]
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 