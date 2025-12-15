"""RAG integration for molecular design suggestions.

This module handles RAG queries for finding molecular modifications
and starting molecules based on literature search.
"""
from typing import List, Dict, Any, Optional
import re

# Import from langchain_tools
try:
    from langchain_tools import retrieve_context, predict_properties
except ImportError:
    retrieve_context = None
    predict_properties = None

from .modifications import MolecularModifier


class RAGIntegration:
    """Handle RAG queries and result processing."""
    
    def __init__(self, use_rag: bool = True, cli_logging: bool = False):
        """Initialize RAG integration.
        
        Args:
            use_rag: Whether to use RAG for suggestions
            cli_logging: Whether to emit CLI logs for RAG actions
        """
        self.use_rag = use_rag
        self.cli_logging = cli_logging
        self.modifier = MolecularModifier()
    
    def find_modifications_with_rag(
        self,
        smiles: str,
        target_properties: Dict[str, float],
        current_score: float
    ) -> List[Dict[str, Any]]:
        """Find molecular modifications using RAG.
        
        Args:
            smiles: Current molecule SMILES
            target_properties: Target property values
            current_score: Current fitness score
        
        Returns:
            List of modification suggestions from RAG
        """
        modifications = []
        
        if not self.use_rag or retrieve_context is None:
            return modifications
        
        try:
            # Get current molecule properties
            if predict_properties:
                current_properties = predict_properties.invoke(smiles)
            else:
                current_properties = {}
            
            # Create search queries for modifications
            search_queries = self.generate_modification_queries(
                smiles, current_properties, target_properties
            )
            
            for query in search_queries[:3]:  # Limit to first 3 queries
                try:
                    # Search RAG for modification strategies
                    rag_results = retrieve_context.invoke(query)
                    
                    if rag_results and len(rag_results) > 0:
                        # Extract modification information from RAG results
                        rag_modifications = self.extract_modifications_from_rag(
                            rag_results, smiles
                        )
                        modifications.extend(rag_modifications)
                        
                except Exception:
                    continue
            
            return modifications
            
        except Exception:
            return modifications
    
    def generate_modification_queries(
        self,
        smiles: str,
        current_properties: Dict[str, float],
        target_properties: Dict[str, float]
    ) -> List[str]:
        """Generate search queries for finding modifications.
        
        Args:
            smiles: Current molecule SMILES
            current_properties: Current property values
            target_properties: Target property values
        
        Returns:
            List of search queries for RAG
        """
        queries = []
        
        # Analyze what properties need improvement
        for prop_name, target_value in target_properties.items():
            if prop_name in current_properties:
                current_value = current_properties[prop_name]
                
                if prop_name == "density" and current_value < target_value:
                    queries.extend([
                        "increase density energetic materials molecular modifications",
                        "high density substituents energetic compounds",
                        "density enhancement energetic materials"
                    ])
                
                elif prop_name == "det_velocity" and current_value < target_value:
                    queries.extend([
                        "increase detonation velocity molecular modifications",
                        "high detonation velocity substituents",
                        "detonation velocity enhancement energetic materials"
                    ])
                
                elif prop_name == "det_pressure" and current_value < target_value:
                    queries.extend([
                        "increase explosion pressure molecular modifications",
                        "high explosion pressure substituents",
                        "explosion pressure enhancement energetic materials"
                    ])
        
        # Add general modification queries
        queries.extend([
            "molecular modifications energetic materials",
            "functional group addition energetic compounds",
            "nitro group addition energetic materials",
            "azido group energetic materials",
            "nitramine synthesis energetic materials"
        ])
        
        return queries
    
    def extract_modifications_from_rag(
        self,
        rag_results: List[Dict[str, Any]],
        current_smiles: str
    ) -> List[Dict[str, Any]]:
        """Extract modification strategies from RAG results.
        
        Args:
            rag_results: RAG search results
            current_smiles: Current molecule SMILES
        
        Returns:
            List of extracted modifications
        """
        modifications = []
        
        for result in rag_results:
            content = result.get('Content', '')
            title = result.get('Title', '')
            authors = result.get('Authors', 'Unknown Authors')
            year = result.get('Year', '')
            
            # Common modification patterns
            modification_patterns = [
                (r'add\s+nitro\s+group', 'nitro_addition'),
                (r'add\s+azido\s+group', 'azido_addition'),
                (r'add\s+nitramine', 'nitramine_addition'),
                (r'add\s+tetrazole', 'tetrazole_addition'),
                (r'remove\s+nitro\s+group', 'nitro_removal'),
                (r'substitute\s+hydrogen', 'hydrogen_substitution'),
            ]
            
            for pattern, mod_type in modification_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    # Try to apply this modification
                    modified_smiles = self.modifier.apply_modification_from_rag(
                        current_smiles, mod_type
                    )
                    if modified_smiles and modified_smiles != current_smiles:
                        modifications.append({
                            'smiles': modified_smiles,
                            'description': f'RAG-suggested {mod_type}',
                            'source': title,
                            'authors': authors,
                            'year': year,
                            'rag_content': content[:100] + "..."
                        })
        
        return modifications
    
    def find_starting_molecule_with_rag(
        self,
        target_properties: Dict[str, float],
        verbose: bool = True
    ) -> Optional[str]:
        """Find starting molecule using RAG.
        
        Args:
            target_properties: Target property values
            verbose: Whether to print progress
        
        Returns:
            SMILES string or None
        """
        if not self.use_rag or retrieve_context is None:
            return None
        
        try:
            # Generate search queries
            search_queries = self.generate_search_queries(target_properties)
            
            if verbose and self.cli_logging:
                print("[RAG] Searching for starting molecules...")
            
            # Search RAG
            all_results = []
            for query in search_queries[:5]:  # Limit queries
                try:
                    results = retrieve_context.invoke(query)
                    if results:
                        all_results.extend(results)
                except Exception:
                    continue
            
            if not all_results:
                return None
            
            # Extract molecules
            molecules = self.extract_molecules_from_rag(all_results)
            
            if not molecules:
                return None
            
            # Return first valid SMILES
            for mol_info in molecules:
                smiles = self.get_smiles_from_molecule_info(mol_info)
                if smiles:
                    if verbose and self.cli_logging:
                        print(f"[RAG] Found starting molecule: {smiles}")
                    return smiles
            
            return None
            
        except Exception:
            return None
    
    def generate_search_queries(
        self,
        target_properties: Dict[str, float]
    ) -> List[str]:
        """Generate search queries for finding starting molecules.
        
        Args:
            target_properties: Target property values
        
        Returns:
            List of search queries
        """
        queries = []
        
        # Property-specific queries
        if "density" in target_properties:
            density = target_properties["density"]
            queries.append(f"high density energetic materials density {density}")
        
        if "det_velocity" in target_properties:
            queries.append("high detonation velocity energetic compounds")
        
        if "det_pressure" in target_properties:
            queries.append("high detonation pressure explosives")
        
        # General queries
        queries.extend([
            "energetic materials database",
            "explosive compounds SMILES",
            "nitro compounds energetic materials"
        ])
        
        return queries
    
    def extract_molecules_from_rag(
        self,
        rag_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract molecule information from RAG results.
        
        Args:
            rag_results: RAG search results
        
        Returns:
            List of molecule information dictionaries
        """
        molecules = []
        
        for result in rag_results:
            content = result.get('Content', '')
            
            # Look for SMILES patterns
            smiles_pattern = r'\b[C-H-N-O-F-Cl-Br-I-S-P-\[\]\(\)=@#\+\-0-9]{6,}\b'
            matches = re.findall(smiles_pattern, content)
            
            for match in matches:
                molecules.append({
                    'smiles': match,
                    'source': result.get('Title', ''),
                    'content': content
                })
        
        return molecules
    
    def get_smiles_from_molecule_info(
        self,
        molecule_info: Dict[str, Any]
    ) -> Optional[str]:
        """Get SMILES from molecule information.
        
        Args:
            molecule_info: Molecule information dictionary
        
        Returns:
            SMILES string or None
        """
        smiles = molecule_info.get('smiles', '')
        
        # Basic validation
        if not smiles or len(smiles) < 3:
            return None
        
        # Try to parse with RDKit
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return smiles
        except:
            pass
        
        return None
