"""
RAG-based modification strategy using Arxiv search and LangChain.
"""

import os
import logging
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .modification_tools import apply_all_modifications, generate_diverse_modifications

logger = logging.getLogger(__name__)


class RAGModificationStrategy:
    """RAG-based molecular modification using Arxiv literature."""
    
    def __init__(self, config):
        """
        Initialize RAG strategy.
        
        Args:
            config: RAGConfig object
        """
        self.config = config
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        
        if config.enable_rag:
            self._initialize_llm()
            self._initialize_embeddings()
    
    def _initialize_llm(self):
        """Initialize ChatOpenAI."""
        try:
            self.llm = ChatOpenAI(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                api_key=self.config.openai_api_key
            )
            logger.info(f"Initialized ChatOpenAI with model {self.config.llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings."""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            logger.info(f"Initialized embeddings with model {self.config.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def generate_rag_query(self, parent_smiles: str, property_gap: Dict[str, float]) -> str:
        """
        Generate an Arxiv search query based on property gaps.
        
        Args:
            parent_smiles: Parent molecule SMILES
            property_gap: Dictionary of property gaps (target - current)
            
        Returns:
            Search query string
        """
        # Analyze which properties need improvement
        needs_increase = []
        needs_decrease = []
        
        for prop, gap in property_gap.items():
            if gap > 0:
                needs_increase.append(prop)
            elif gap < 0:
                needs_decrease.append(prop)
        
        # Build query based on properties
        query_parts = ["energetic materials", "high energy density"]
        
        if 'Density' in needs_increase:
            query_parts.append("increase density")
        if 'Det Velocity' in needs_increase or 'Det Pressure' in needs_increase:
            query_parts.append("detonation performance")
            query_parts.append("nitro compounds")
        if 'Hf solid' in needs_increase:
            query_parts.append("formation enthalpy")
        
        # Add structural keywords
        query_parts.extend(["molecular design", "functional groups"])
        
        query = " ".join(query_parts)
        logger.info(f"Generated RAG query: {query}")
        return query
    
    def search_arxiv_papers(self, query: str) -> List[Document]:
        """
        Search Arxiv for relevant papers.
        
        Args:
            query: Search query
            
        Returns:
            List of Document objects
        """
        try:
            loader = ArxivLoader(
                query=query,
                load_max_docs=self.config.arxiv_max_results
            )
            docs = loader.load()
            logger.info(f"Retrieved {len(docs)} papers from Arxiv")
            return docs
        except Exception as e:
            logger.error(f"Error searching Arxiv: {e}")
            return []
    
    def chunk_and_embed_papers(self, papers: List[Document]) -> Optional[FAISS]:
        """
        Chunk papers and create vector store using FAISS.
        
        Args:
            papers: List of paper documents
            
        Returns:
            FAISS vectorstore
        """
        if not papers:
            return None
        
        try:
            # Use RecursiveCharacterTextSplitter for consistent chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Split documents
            chunks = text_splitter.split_documents(papers)
            logger.info(f"Split papers into {len(chunks)} chunks")
            
            # Create FAISS vector store (much more stable than ChromaDB)
            try:
                logger.info(f"Creating FAISS vectorstore with {len(chunks)} chunks...")
                vectorstore = FAISS.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                logger.info(f"Successfully created FAISS vectorstore")
                return vectorstore
            except Exception as ve:
                logger.error(f"FAISS creation failed: {ve}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        except Exception as e:
            logger.error(f"Error chunking and embedding papers: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def query_rag_system(self, question: str, vectorstore: FAISS, k: int = 5) -> str:
        """
        Query the RAG system for relevant context.
        
        Args:
            question: Question to ask
            vectorstore: Vector store to query
            k: Number of documents to retrieve
            
        Returns:
            Retrieved context as string
        """
        try:
            docs = vectorstore.similarity_search(question, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context
        except Exception as e:
            logger.error(f"Error querying vectorstore: {e}")
            return ""
    
    def extract_modification_strategies(self, parent_smiles: str, 
                                       property_gap: Dict[str, float],
                                       context: str) -> List[str]:
        """
        Use LLM to extract modification strategies from context.
        
        Args:
            parent_smiles: Parent molecule SMILES
            property_gap: Property gaps
            context: Retrieved context from papers
            
        Returns:
            List of suggested SMILES modifications
        """
        # Build prompt
        gap_str = ", ".join([f"{k}: {v:+.2f}" for k, v in property_gap.items() if abs(v) > 0.01])
        
        prompt = f"""You are an expert in energetic materials chemistry. Given a parent molecule and desired property changes, suggest specific molecular modifications.

Parent Molecule (SMILES): {parent_smiles}

Property Gaps (target - current):
{gap_str}

Relevant Literature Context:
{context[:2000]}  

Based on the literature and chemical principles, suggest {self.config.max_modifications_per_call} specific modifications to the parent molecule. 
Focus on functional group additions, substitutions, or structural changes that would improve the target properties.

Provide your answer as a numbered list of modification strategies in plain text (e.g., "1. Add nitro group to aromatic ring", "2. Replace hydrogen with azido group").
Be specific and chemistry-focused, but concise."""

        try:
            response = self.llm.invoke(prompt)
            strategies_text = response.content
            logger.info(f"LLM suggested strategies:\n{strategies_text}")
            
            # Parse strategies (simple numbered list extraction)
            strategies = []
            for line in strategies_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering
                    strategy = line.lstrip('0123456789.-•) ').strip()
                    if strategy:
                        strategies.append(strategy)
            
            return strategies
        
        except Exception as e:
            logger.error(f"Error extracting strategies from LLM: {e}")
            return []
    
    def apply_rag_modifications(self, smiles: str, strategies: List[str], target_count: int = 10) -> List[str]:
        """
        Apply RAG-suggested modifications to molecule using SMARTS reactions.
        
        Parses LLM strategies to identify relevant chemical modifications,
        maps them to SMARTS patterns, applies reactions, and filters out
        modifications that result in multiple molecules (fragmentation).
        
        Args:
            smiles: Parent SMILES
            strategies: List of modification strategies from LLM
            target_count: Target number of modifications to return
            
        Returns:
            List of modified SMILES (up to target_count, single molecules only)
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        logger.info(f"Applying RAG strategies to {smiles} (target: {target_count} modifications)")
        logger.info(f"Strategies received: {strategies}")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return []
        
        # Define SMARTS reactions for different modification strategies
        # Format: (keywords, reaction_smarts_list)
        strategy_reactions = {
            'nitro': [
                # Add nitro to aromatic carbon
                ('[c:1][H]>>[c:1][N+](=O)[O-]', 'Add NO2 to aromatic C'),
                # Add nitro to aliphatic carbon
                ('[C:1][H]>>[C:1][N+](=O)[O-]', 'Add NO2 to aliphatic C'),
            ],
            'azido': [
                # Add azido group to aromatic carbon
                ('[c:1][H]>>[c:1]N=[N+]=[N-]', 'Add N3 to aromatic C'),
                # Add azido group to aliphatic carbon
                ('[C:1][H]>>[C:1]N=[N+]=[N-]', 'Add N3 to aliphatic C'),
            ],
            'amino': [
                # Add amino group
                ('[c:1][H]>>[c:1]N', 'Add NH2 to aromatic C'),
                ('[C:1][H]>>[C:1]N', 'Add NH2 to aliphatic C'),
            ],
            'hydroxyl': [
                # Add hydroxyl group
                ('[c:1][H]>>[c:1]O', 'Add OH to aromatic C'),
                ('[C:1][H]>>[C:1]O', 'Add OH to aliphatic C'),
            ],
            'cyano': [
                # Add cyano group
                ('[c:1][H]>>[c:1]C#N', 'Add CN to aromatic C'),
                ('[C:1][H]>>[C:1]C#N', 'Add CN to aliphatic C'),
            ],
            'fluoro': [
                # Fluorination
                ('[c:1][H]>>[c:1]F', 'Add F to aromatic C'),
                ('[C:1][H]>>[C:1]F', 'Add F to aliphatic C'),
            ],
            'chloro': [
                # Chlorination
                ('[c:1][H]>>[c:1]Cl', 'Add Cl to aromatic C'),
                ('[C:1][H]>>[C:1]Cl', 'Add Cl to aliphatic C'),
            ],
            'methyl': [
                # Methylation
                ('[c:1][H]>>[c:1]C', 'Add CH3 to aromatic C'),
                ('[C:1][H]>>[C:1]C', 'Add CH3 to aliphatic C'),
            ],
            'tetrazole': [
                # Replace CN with tetrazole (common energetic transformation)
                ('[C:1]#N>>[C:1]c1nnn[nH]1', 'Replace CN with tetrazole'),
            ],
            'triazole': [
                # Add triazole ring via azide-alkyne cycloaddition conceptually
                ('[c:1][H]>>[c:1]c1cn[nH]n1', 'Add triazole to aromatic C'),
            ],
            'furazan': [
                # Add furazan (1,2,5-oxadiazole) - energetic heterocycle
                ('[c:1][H]>>[c:1]c1nonc1', 'Add furazan ring'),
            ],
            'nitramine': [
                # Convert amine to nitramine
                ('[N:1]([H])[H]>>[N:1]([H])[N+](=O)[O-]', 'Convert NH2 to NH-NO2'),
            ],
            'oxidation': [
                # N-oxidation (for pyridine-like nitrogens)
                ('[n:1]>>[n+:1][O-]', 'N-oxide formation'),
            ],
            'ring': [
                # General ring modifications are handled via the simple rings
                ('[C:1][C:2][C:3][C:4]>>[C:1]1[C:2][C:3][C:4]1', 'Cyclization'),
            ],
            'density': [
                # Heavy atoms to increase density
                ('[c:1][H]>>[c:1][N+](=O)[O-]', 'Add NO2 for density'),
                ('[c:1][H]>>[c:1]N=[N+]=[N-]', 'Add N3 for density'),
            ],
            'detonation': [
                # Groups that improve detonation performance
                ('[c:1][H]>>[c:1][N+](=O)[O-]', 'Add NO2 for detonation'),
                ('[N:1]([H])[H]>>[N:1]([H])[N+](=O)[O-]', 'Nitramine for detonation'),
            ],
            'energy': [
                # High-energy groups
                ('[c:1][H]>>[c:1][N+](=O)[O-]', 'Add NO2 for energy'),
                ('[c:1][H]>>[c:1]N=[N+]=[N-]', 'Add N3 for energy'),
            ],
        }
        
        # Collect all reactions to apply based on strategies
        reactions_to_apply = []
        
        for strategy in strategies:
            strategy_lower = strategy.lower()
            
            # Check each keyword category
            for keyword, reactions in strategy_reactions.items():
                if keyword in strategy_lower:
                    for rxn_smarts, rxn_name in reactions:
                        reactions_to_apply.append((rxn_smarts, f"{rxn_name} (from: {strategy[:30]}...)"))
        
        # If no specific strategies matched, use general energetic modifications
        if not reactions_to_apply:
            logger.info("No specific strategies matched, using general energetic modifications")
            for keyword in ['nitro', 'azido', 'amino']:
                for rxn_smarts, rxn_name in strategy_reactions[keyword]:
                    reactions_to_apply.append((rxn_smarts, rxn_name))
        
        logger.info(f"Applying {len(reactions_to_apply)} reaction patterns from strategies")
        
        # Apply reactions and collect results
        all_modifications = set()
        
        for rxn_smarts, rxn_name in reactions_to_apply:
            try:
                rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                if rxn is None:
                    logger.debug(f"Failed to parse reaction: {rxn_smarts}")
                    continue
                
                products = rxn.RunReactants((mol,))
                
                for product_tuple in products:
                    for product in product_tuple:
                        try:
                            Chem.SanitizeMol(product)
                            new_smiles = Chem.MolToSmiles(product)
                            
                            # Filter out modifications with multiple molecules (fragmentation)
                            if '.' in new_smiles:
                                logger.debug(f"Skipping fragmented product: {new_smiles}")
                                continue
                            
                            # Skip if same as parent
                            if new_smiles == smiles:
                                continue
                            
                            # Validate the molecule is reasonable
                            if Chem.MolFromSmiles(new_smiles) is not None:
                                all_modifications.add(new_smiles)
                                
                        except Exception as e:
                            logger.debug(f"Failed to process product from {rxn_name}: {e}")
                            continue
                            
            except Exception as e:
                logger.debug(f"Reaction {rxn_name} failed: {e}")
                continue
        
        logger.info(f"Generated {len(all_modifications)} modifications from RAG strategies")
        
        # If we don't have enough, supplement with diverse modifications
        if len(all_modifications) < target_count:
            logger.info(f"Supplementing with diverse modifications (have {len(all_modifications)}, need {target_count})")
            diverse_mods = generate_diverse_modifications(smiles, target_count=target_count - len(all_modifications))
            
            # Filter diverse mods for single molecules only
            for mod in diverse_mods:
                if '.' not in mod and mod != smiles:
                    all_modifications.add(mod)
        
        result = list(all_modifications)[:target_count]
        logger.info(f"Returning {len(result)} total modifications")
        return result
    
    def generate_multiple_queries(self, property_gap: Dict[str, float]) -> List[str]:
        """
        Generate multiple diverse queries for different modification strategies.
        Property-focused queries based on what needs to be improved.
        
        Returns:
            List of different query strings for varied RAG results
        """
        queries = []
        
        # Density-focused queries
        if property_gap.get('Density', 0) > 0:
            queries.append("increasing density of energetic materials molecular design")
            # queries.append("high density explosives crystal packing optimization")
            # queries.append("dense energetic compounds nitro groups molecular weight")
            # queries.append("improving density in nitrogen-rich heterocycles")
        elif property_gap.get('Density', 0) < 0:
            queries.append("decreasing density of energetic materials")
            # queries.append("lightweight explosives low density propellants")
            # queries.append("reducing molecular weight in energetic compounds")
        
        # Detonation velocity queries
        if property_gap.get('Det Velocity', 0) > 0:
            queries.append("increasing detonation velocity of energetic materials")
            # queries.append("high velocity explosives performance enhancement")
            # queries.append("improving detonation speed nitro compounds")
        elif property_gap.get('Det Velocity', 0) < 0:
            queries.append("decreasing detonation velocity controlled explosives")
            # queries.append("low velocity energetic materials insensitive munitions")
        
        # Detonation pressure queries
        if property_gap.get('Det Pressure', 0) > 0:
            queries.append("increasing detonation pressure of explosives")
            # queries.append("high pressure energetic materials performance")
            # queries.append("improving brisance detonation pressure synthesis")
        elif property_gap.get('Det Pressure', 0) < 0:
            queries.append("decreasing detonation pressure energetic materials")
            # queries.append("low pressure explosives reduced sensitivity")
        
        # Heat of formation queries
        if property_gap.get('Hf solid', 0) > 0:
            queries.append("increasing heat of formation energetic materials")
            # queries.append("positive enthalpy of formation high energy compounds")
            # queries.append("nitrogen-rich compounds high formation enthalpy")
            # queries.append("endothermic energetic materials polynitrogen synthesis")
        elif property_gap.get('Hf solid', 0) < 0:
            queries.append("decreasing heat of formation energetic materials")
            # queries.append("stable explosives low formation enthalpy")
            # queries.append("thermally stable energetic compounds synthesis")
        
        # Structural modification queries (based on property needs)
        # if any(gap > 0 for gap in property_gap.values()):
            # queries.append("adding nitro groups to improve energetic performance")
            # queries.append("azido tetrazole functional groups high energy")
            # queries.append("heterocyclic energetic compounds triazole pyrazole")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        logger.info(f"Generated {len(unique_queries)} property-focused RAG queries")
        return unique_queries
    
    def rag_modification_strategy(self, smiles: str, property_gap: Dict[str, float], target_count: int = 10) -> List[str]:
        """
        Multi-query RAG-based modification strategy.
        
        Makes multiple RAG calls with different queries to maximize candidate diversity.
        
        Args:
            smiles: Parent molecule SMILES
            property_gap: Property gaps (target - current)
            target_count: Target number of modifications to generate
            
        Returns:
            List of modified SMILES (attempts to generate target_count modifications)
        """
        if not self.config.enable_rag:
            logger.info("RAG disabled, returning empty list")
            return []
        
        all_modifications = set()
        
        try:
            # Generate multiple diverse queries
            queries = self.generate_multiple_queries(property_gap)
            
            # Try each query until we have enough candidates
            for i, query in enumerate(queries):
                if len(all_modifications) >= target_count:
                    logger.info(f"Reached target count after {i+1} queries")
                    break
                    
                logger.info(f"RAG Query {i+1}/{len(queries)}: {query[:50]}...")
                
                try:
                    # Search Arxiv with this query
                    papers = self.search_arxiv_papers(query)
                    if not papers:
                        continue
                    
                    # Chunk and embed
                    vectorstore = self.chunk_and_embed_papers(papers)
                    if vectorstore is None:
                        continue
                    
                    # Query for relevant modifications
                    question = f"What chemical modifications can improve {', '.join(property_gap.keys())} in energetic molecules?"
                    context = self.query_rag_system(question, vectorstore)
                    
                    if not context:
                        continue
                    
                    # Extract strategies from this context
                    strategies = self.extract_modification_strategies(smiles, property_gap, context)
                    
                    if strategies:
                        # Apply modifications
                        new_mods = self.apply_rag_modifications(smiles, strategies, target_count=target_count)
                        
                        # Add new unique modifications
                        before_count = len(all_modifications)
                        all_modifications.update(new_mods)
                        added = len(all_modifications) - before_count
                        logger.info(f"Query {i+1} added {added} new modifications (total: {len(all_modifications)})")
                        
                except Exception as qe:
                    logger.warning(f"Query {i+1} failed: {qe}")
                    continue
            
            logger.info(f"RAG multi-query generated {len(all_modifications)} total modifications (target: {target_count})")
            return list(all_modifications)
        
        except Exception as e:
            logger.error(f"RAG multi-query strategy failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


def default_modification_strategy(smiles: str, property_gap: Dict[str, float], target_count: int = 20) -> List[str]:
    """
    Default heuristic-based modification strategy (fallback).
    
    Args:
        smiles: Parent molecule SMILES
        property_gap: Property gaps (target - current)
        target_count: Target number of modifications (default 20)
        
    Returns:
        List of modified SMILES
    """
    logger.info(f"Using default modification strategy for {smiles} (target: {target_count})")
    
    # Use the diverse modification generator for maximum coverage
    modifications = generate_diverse_modifications(smiles, target_count=target_count * 2)
    
    # If we still don't have enough, also try specific strategies based on property gaps
    if len(modifications) < target_count:
        from .modification_tools import (
            addition_modification,
            substitution_modification,
            ring_modification
        )
        
        additional = []
        
        # Strategy based on property gaps
        if property_gap.get('Density', 0) > 0:
            # Increase density: add heavy atoms, fuse rings
            additional.extend(ring_modification(smiles))
            additional.extend(addition_modification(smiles, ['[N+](=O)[O-]']))
        
        if property_gap.get('Det Velocity', 0) > 0 or property_gap.get('Det Pressure', 0) > 0:
            # Increase detonation properties: add energetic groups
            additional.extend(addition_modification(smiles, ['[N+](=O)[O-]', 'N=[N+]=[N-]']))
            additional.extend(substitution_modification(smiles))
        
        # Add to existing modifications
        modifications.extend(additional)
    
    # Remove duplicates
    modifications = list(set(modifications))
    modifications = [m for m in modifications if m != smiles]
    
    logger.info(f"Default strategy generated {len(modifications)} candidates (target: {target_count})")
    return modifications[:target_count * 2]  # Return up to 2x target for filtering headroom
